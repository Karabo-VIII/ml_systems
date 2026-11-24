"""Crypto-layer audit regressions (contract 2.2.0)."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

import pipeline.chunked as chunked
import pipeline.io as pio
from pipeline.bars import DollarAccumulator, dollar_bars, trailing_threshold
from pipeline.config import Splits, lag_ms, load_lags
from pipeline.constants import CONTRACT_VERSION, DAY_MS
from pipeline.io import atomic_write_parquet, file_sha256
from pipeline.orchestrator import RunConfig, run
from pipeline.partitions import partition_fresh
from fixtures import small_windows, write_bronze_day
from test_resume import _day


def _trades(ts_ms: list[int], usd: list[float], start_id: int = 1) -> pl.DataFrame:
    n = len(ts_ms)
    return pl.DataFrame({
        "agg_trade_id": list(range(start_id, start_id + n)),
        "ts_us": [t * 1000 for t in ts_ms],
        "ts_ms": ts_ms,
        "price": [1.0] * n,
        "qty": [float(u) for u in usd],
        "is_buyer_maker": [False] * n,
    })


def _ms(y, m, d, h=0, mi=0, s=0) -> int:
    return int(datetime(y, m, d, h, mi, s, tzinfo=timezone.utc).timestamp() * 1000)


# --- event bars -----------------------------------------------------------------

def test_crossing_trade_closes_its_bar_and_every_bar_reaches_the_threshold():
    base = _ms(2024, 6, 1)
    trades = _trades([base + i * 1000 for i in range(10)], [40.0] * 10)
    bars, excluded = dollar_bars(trades, 100.0, "BTCUSDT")
    # 40+40+40 = 120 closes on the third trade; the fourth starts a new bar.
    assert bars["n_trades"].to_list() == [3, 3, 3]
    assert bars["volume_usd"].to_list() == [120.0, 120.0, 120.0]
    assert bars["ts_ms"].to_list() == [base + 2000, base + 5000, base + 8000]
    assert excluded == 1  # the open tail is withheld, never emitted as a short bar
    assert bars["volume_usd"].min() >= 100.0


def test_dollar_bar_ids_are_dense_and_unique():
    base = _ms(2024, 6, 1)
    rng = np.random.default_rng(3)
    trades = _trades([base + i * 500 for i in range(500)], list(rng.uniform(1, 90, 500)))
    bars, _ = dollar_bars(trades, 200.0, "ETHUSDT")
    assert bars["bar_id"].to_list() == list(range(bars.height))
    assert (bars["volume_usd"] >= 200.0 - 1e-9).all()


def test_trailing_threshold_is_median_over_target_with_bootstrap():
    assert trailing_threshold([], 288, 777.0) == 777.0
    assert trailing_threshold([288.0, 576.0, 2880.0], 288, 777.0) == 2.0
    # junk in the history never produces a zero or NaN threshold
    assert trailing_threshold([0.0, float("nan")], 288, 5.0) == 5.0


def test_adaptive_threshold_is_causal():
    """Day D's trades can not move day D's own threshold; only earlier days do."""
    days = [_ms(2024, 6, d) for d in (1, 2, 3)]

    def run_acc(day3_scale: float) -> DollarAccumulator:
        acc = DollarAccumulator(10.0, target_bars_per_day=100.0, window_days=30)
        for i, base in enumerate(days):
            scale = day3_scale if i == 2 else 1.0
            acc.add(_trades([base + k * 1000 for k in range(200)], [scale * 50.0] * 200))
        return acc

    a, b = run_acc(1.0), run_acc(1000.0)
    # day 1 -> bootstrap; day 2 -> notional(day1)/100 = 100; day 3 -> median(day1, day2)/100 = 100
    assert a.history[-2:] == [10000.0, 10000.0]
    assert a.cur_day_threshold == 100.0
    assert b.cur_day_threshold == 100.0  # day 3's own flood did not change day 3's size
    assert a.history == b.history


def test_accumulator_state_roundtrip_matches_one_shot_across_midnight_and_months():
    rng = np.random.default_rng(5)
    parts = []
    tid = 1
    for d in (_ms(2024, 5, 31), _ms(2024, 6, 1), _ms(2024, 6, 2)):
        ts = [d + k * 1500 for k in range(3000)]
        parts.append(_trades(ts, list(rng.uniform(1, 400, 3000)), tid))
        tid += 3000

    def make():
        return DollarAccumulator(300.0, target_bars_per_day=50.0, window_days=30)

    whole = make()
    for p in parts:
        whole.add(p)
    expect, _ = whole.emit_completed("BTCUSDT")

    acc = make()
    got = []
    for i, p in enumerate(parts):
        acc.add(p)
        bars, _ = acc.emit_completed("BTCUSDT")
        got.append(bars)
        meta, frag = acc.state()
        meta = json.loads(json.dumps(meta))  # exactly what is written to _state/
        acc = DollarAccumulator.restore(meta, frag)
    got = pl.concat(got)
    assert got.height == expect.height
    assert got["ts_ms"].to_list() == expect["ts_ms"].to_list()
    assert got["bar_id"].to_list() == expect["bar_id"].to_list()
    assert np.allclose(got["volume_usd"].to_numpy(), expect["volume_usd"].to_numpy())


# --- freshness / lags / klines --------------------------------------------------

def test_partition_fresh_rejects_a_file_from_an_older_contract(tmp_path, monkeypatch):
    path = tmp_path / "x.parquet"
    params = {"a": 1}
    df = pl.DataFrame({"ts_ms": [1, 2]})
    monkeypatch.setattr(pio, "CONTRACT_VERSION", "2.1.0")
    atomic_write_parquet(df, path, schema_name="t", params=params)
    monkeypatch.undo()
    assert not partition_fresh(path, params)
    atomic_write_parquet(df, path, schema_name="t", params=params)
    assert partition_fresh(path, params)


def test_open_interest_lag_covers_the_five_minute_window_and_slack():
    # Vision metrics stamp the START of a 5 minute window; the value describes T+5m.
    assert lag_ms("open_interest") >= 600_000
    assert load_lags()["etf"] >= 2 * DAY_MS
    assert load_lags()["wikipedia"] >= 2 * DAY_MS


def test_norm_oi_change_is_unchanged_by_future_open_interest():
    from pipeline.align import asof_join_knowable
    from pipeline.bars import trades_to_time
    from pipeline.features import compute_physics
    from fixtures import make_trades

    start = datetime(2024, 6, 1, 0, 0, 30)
    bars = trades_to_time(make_trades(400, start=start, step=timedelta(minutes=1)), 60_000, "BTCUSDT", "time_1m")
    w = small_windows(z=10, frac_diff=15, hurst=10, ma=10, perm_entropy=10, kurtosis=8, kyle=8, yz=8, cs_block=4, vol_cluster=8)
    ts = bars["ts_ms"].to_list()
    oi = [1e6 + 100.0 * i for i in range(len(ts))]
    cut = 250

    def feat(values):
        m = pl.DataFrame({"ts_ms": ts, "oi_value": values})
        j = asof_join_knowable(bars, m, ["oi_value"], lag_ms("open_interest"))
        return compute_physics(j, w, include_oi=True)

    clean = feat(oi)
    pert = feat(oi[:cut] + [v * 9 for v in oi[cut:]])
    horizon = cut + lag_ms("open_interest") // 60_000  # first bar that may see a perturbed print
    for col in ("oi_value", "norm_oi_change", "norm_oi_price_divergence"):
        a, b = clean[col].to_numpy()[:horizon], pert[col].to_numpy()[:horizon]
        assert np.allclose(a, b, equal_nan=True), col
    assert clean["norm_oi_change"].drop_nulls().len() > 0


def test_kline_parser_normalises_microsecond_open_times():
    from pipeline.fetch_crypto import parse_klines

    ms = _ms(2025, 1, 2)
    rows = "\n".join(
        f"{t},1,2,0.5,1.5,10,{t + 59_999},15,3,4,5,0" for t in (ms * 1000, (ms + 60_000) * 1000)
    ).encode()
    out = parse_klines(rows)
    assert out["ts_open_ms"].to_list() == [ms, ms + 60_000]
    assert out["ts_ms"].to_list() == [ms + 60_000, ms + 120_000]


# --- pipeline level -------------------------------------------------------------

def _cfg(tmp_path, d1, d2, **extra):
    kw = dict(
        start=d1, end=d2, assets=["BTCUSDT"], stages=["bars", "features", "dataset"],
        bar_types=["time_1m", "time_1h", "time_1d", "dollar"], data_root=tmp_path,
        windows=small_windows(), threshold_overrides={"BTCUSDT": 5000.0},
    )
    kw.update(extra)
    return RunConfig(**kw)


def _seed(tmp_path, days):
    for i, d in enumerate(days):
        write_bronze_day(tmp_path, "BTCUSDT", d, _day(i + 1, 100.0 + i, d))


def _gold_hashes(root):
    return {str(p.relative_to(root)): file_sha256(p) for p in (root / "gold").rglob("*.parquet")}


def test_gold_manifest_fields_are_populated(tmp_path):
    d1, d2 = date(2024, 5, 31), date(2024, 6, 1)
    _seed(tmp_path, [d1, d2])
    assert run(_cfg(tmp_path, d1, d2)) == 0
    man = json.loads((tmp_path / "gold" / "time_1h" / "BTCUSDT_2024-06.manifest.json").read_text())
    assert man["contract_version"] == CONTRACT_VERSION
    for key in ("params_hash", "inputs", "lags_applied", "split_boundaries", "null_by_design", "embargo_bars", "n_rows"):
        assert man.get(key) not in (None, "", []), key
    assert man["split_boundaries"]["train_end"]
    assert {i["family"] for i in man["inputs"]} >= {"funding", "daily_panel"}
    assert man["lags_applied"]
    for family, lag in man["lags_applied"].items():
        assert lag == lag_ms(family)


def test_gold_is_idempotent_and_older_contract_partitions_rebuild(tmp_path):
    d1, d2 = date(2024, 5, 31), date(2024, 6, 1)
    _seed(tmp_path, [d1, d2])
    assert run(_cfg(tmp_path, d1, d2)) == 0
    first = _gold_hashes(tmp_path)
    assert first
    assert run(_cfg(tmp_path, d1, d2, force=True)) == 0
    assert _gold_hashes(tmp_path) == first


def test_gold_month_is_rebuilt_when_the_next_month_arrives(tmp_path):
    """Targets look 64 bars ahead, so the last bars of May change once June exists."""
    d1, d2 = date(2024, 5, 31), date(2024, 6, 1)
    _seed(tmp_path, [d1])
    assert run(_cfg(tmp_path, d1, d1)) == 0
    path = tmp_path / "gold" / "time_1m" / "BTCUSDT_2024-05.parquet"
    before = pl.read_parquet(path)
    assert before["target_return_64"].null_count() >= 64
    write_bronze_day(tmp_path, "BTCUSDT", d2, _day(2, 101.0, d2))
    assert run(_cfg(tmp_path, d1, d2)) == 0
    after = pl.read_parquet(path)
    assert after["target_return_64"].null_count() < before["target_return_64"].null_count()


def test_bs_basis_is_present_on_every_cadence(tmp_path):
    d1, d2 = date(2024, 5, 31), date(2024, 6, 1)
    _seed(tmp_path, [d1, d2])
    from pipeline.paths import perp_1d_path as perp_path
    rows = []
    for d in (d1, d2):
        o = _ms(d.year, d.month, d.day)
        rows.append({"ts_open_ms": o, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
                     "volume": 1.0, "quote_volume": 100.0, "n_trades": 1, "ts_ms": o + DAY_MS})
        p = perp_path(tmp_path, "BTCUSDT", d)
        atomic_write_parquet(pl.DataFrame([rows[-1]]), p, schema_name="perp_1d_v1", params={"d": d.isoformat()})
    assert run(_cfg(tmp_path, d1, d2)) == 0
    for bt in ("time_1m", "time_1h", "time_1d"):
        gold = pl.read_parquet(tmp_path / "gold" / bt / "BTCUSDT_2024-06.parquet")
        assert "bs_basis" in gold.columns, bt
    assert pl.read_parquet(tmp_path / "gold" / "time_1h" / "BTCUSDT_2024-06.parquet")["bs_basis"].drop_nulls().len() > 0


def test_splits_and_embargo_hold_across_month_partitions(tmp_path):
    """Daily bars: the embargo each side of a split edge, across three partitions.

    The edge is 2024-05-10, so the tail of train reaches back into the April
    partition, further from the edge than the target horizon. Under 2.3.0 the
    gold frame of April was padded only by the horizon, never saw the edge, and
    left that tail labelled ``train``.
    """
    from pipeline.config import embargo_bars_for
    d1, d2 = date(2024, 3, 20), date(2024, 6, 5)
    days = [d1 + timedelta(days=i) for i in range((d2 - d1).days + 1)]
    _seed(tmp_path, days)
    splits = Splits(train_end=date(2024, 5, 10), val_end=date(2030, 1, 1), oos_end=date(2031, 1, 1),
                    unseen_start=date(2031, 1, 1), live_start=date(2032, 1, 1))
    windows = small_windows(
        z=3, fast=3, vol_cluster=3, frac_diff=4, hurst=4, kurtosis=3, efficiency=2, perm_entropy=4,
        kyle=3, yz=3, cs_block=2, funding_mom=3, hawkes_span=3, ma=3, slow_span=3, horizons=(1, 2))
    cfg = _cfg(tmp_path, d1, d2, splits=splits, bar_types=["time_1d"], windows=windows)
    assert run(cfg) == 0
    gold = tmp_path / "gold" / "time_1d"
    n = json.loads((gold / "BTCUSDT_2024-05.manifest.json").read_text())["embargo_bars"]
    # Input window 17 (the 16-bar return) + 1, then the 3-bar z-score; floor 7 days.
    assert n == embargo_bars_for("time_1d", windows, calendar_floor_days=7) == 21
    both = pl.concat([pl.read_parquet(p) for p in sorted(gold.glob("BTCUSDT_*.parquet"))]).sort("ts_ms")
    edge = _ms(2024, 5, 10)
    pre = [l for t, l in zip(both["ts_ms"], both["split"]) if t < edge]
    post = [l for t, l in zip(both["ts_ms"], both["split"]) if t >= edge]
    assert pre[-n:] == ["embargo"] * n          # tail of train, spilling into April
    assert set(pre[:-n]) == {"train"} and len(pre) > n
    assert post[:n] == ["embargo"] * n          # head of val
    assert post[n] == "val"
    april = pl.read_parquet(gold / "BTCUSDT_2024-04.parquet")
    assert "embargo" in april["split"].to_list()


# --- dollar fabric: resume, invalidation, no double counting ---------------------

def _fabric(root):
    out = {}
    for sub in ("silver", "gold"):
        for p in (root / sub).rglob("*.parquet"):
            out[str(p.relative_to(root))] = file_sha256(p)
    return out


def _twin_roots(tmp_path):
    d1, d2 = date(2024, 5, 31), date(2024, 6, 1)
    clean, broken = tmp_path / "clean", tmp_path / "broken"
    for root in (clean, broken):
        _seed(root, [d1, d2])
    assert run(_cfg(clean, d1, d2)) == 0
    return d1, d2, clean, broken


def test_crash_after_dollar_partition_before_state_save_resumes_exactly(tmp_path, monkeypatch):
    d1, d2, clean, broken = _twin_roots(tmp_path)
    real = chunked.save_dollar_state

    def crash(root, asset, acc, through):
        if through == "2024-06":
            raise KeyboardInterrupt()
        return real(root, asset, acc, through)

    monkeypatch.setattr(chunked, "save_dollar_state", crash)
    with pytest.raises(KeyboardInterrupt):
        run(_cfg(broken, d1, d2))
    monkeypatch.setattr(chunked, "save_dollar_state", real)
    assert run(_cfg(broken, d1, d2)) == 0
    assert _fabric(broken) == _fabric(clean)


def test_torn_state_write_is_detected_and_rebuilt(tmp_path, monkeypatch):
    d1, d2, clean, broken = _twin_roots(tmp_path)
    real = chunked.atomic_write_json

    def crash(path, payload):
        if "dollar_" in str(path) and payload.get("through") == "2024-06":
            raise KeyboardInterrupt()
        return real(path, payload)

    monkeypatch.setattr(chunked, "atomic_write_json", crash)
    with pytest.raises(KeyboardInterrupt):
        run(_cfg(broken, d1, d2))
    monkeypatch.setattr(chunked, "atomic_write_json", real)
    # fragments are June's, the JSON is still May's: the state must be refused, not trusted
    spec = chunked.dollar_spec_for(_cfg(broken, d1, d2), "BTCUSDT")
    assert chunked.read_dollar_state(broken, "BTCUSDT", spec) is None or \
        chunked.read_dollar_state(broken, "BTCUSDT", spec).through == "2024-05"
    assert run(_cfg(broken, d1, d2)) == 0
    assert _fabric(broken) == _fabric(clean)


def test_interrupt_after_first_bars_month_resumes_dollar_chain(tmp_path, monkeypatch):
    d1, d2, clean, broken = _twin_roots(tmp_path)
    real = chunked.write_bars_month
    calls = {"n": 0}

    def boom(*a, **k):
        out = real(*a, **k)
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt()
        return out

    import pipeline.orchestrator as orch
    monkeypatch.setattr(orch, "write_bars_month", boom)
    with pytest.raises(KeyboardInterrupt):
        run(_cfg(broken, d1, d2))
    monkeypatch.setattr(orch, "write_bars_month", real)
    assert run(_cfg(broken, d1, d2)) == 0
    assert _fabric(broken) == _fabric(clean)


def test_a_different_start_invalidates_the_dollar_chain(tmp_path):
    d1, d2 = date(2024, 5, 30), date(2024, 6, 1)
    _seed(tmp_path, [d1, date(2024, 5, 31), d2])
    cfg = _cfg(tmp_path, d1, d2)
    assert run(cfg) == 0
    months = [(2024, 5), (2024, 6)]
    f, _ = chunked.plan_dollar(cfg, "BTCUSDT", months, d1)
    assert f == 2
    f2, acc = chunked.plan_dollar(cfg, "BTCUSDT", months, date(2024, 5, 31))
    assert f2 == 0 and acc.next_id == 0 and acc.through is None


def test_rebuilding_only_time_bars_never_feeds_cached_dollar_months(tmp_path):
    d1, d2 = date(2024, 5, 31), date(2024, 6, 1)
    _seed(tmp_path, [d1, d2])
    assert run(_cfg(tmp_path, d1, d2)) == 0
    dollar = {p.name: file_sha256(p) for p in (tmp_path / "silver" / "bars" / "dollar").rglob("*.parquet")}
    state_before = (tmp_path / "silver" / "bars" / "_state" / "dollar_BTCUSDT.json").read_text()
    for p in (tmp_path / "silver" / "bars" / "time_1h").rglob("*"):
        if p.is_file():
            p.unlink()
    assert run(_cfg(tmp_path, d1, d2)) == 0
    after = {p.name: file_sha256(p) for p in (tmp_path / "silver" / "bars" / "dollar").rglob("*.parquet")}
    assert after == dollar
    assert (tmp_path / "silver" / "bars" / "_state" / "dollar_BTCUSDT.json").read_text() == state_before


# --- validators -----------------------------------------------------------------

def _one_run(tmp_path):
    d1, d2 = date(2024, 5, 31), date(2024, 6, 1)
    _seed(tmp_path, [d1, d2])
    assert run(_cfg(tmp_path, d1, d2)) == 0
    return d1, d2


def _write_klines_from(root, bars_1m: pl.DataFrame, asset="BTCUSDT"):
    from pipeline.paths import klines_1m_path
    k = bars_1m.select([
        "ts_open_ms", "open", "high", "low", "close", "volume",
        pl.col("volume_usd").alias("quote_volume"), "n_trades", "ts_ms",
    ]).with_columns(
        pl.lit(0.0).alias("taker_buy_base"), pl.lit(0.0).alias("taker_buy_quote"),
    )
    k = k.with_columns(pl.from_epoch(pl.col("ts_open_ms"), time_unit="ms").dt.date().alias("_d"))
    for (d,), part in k.group_by("_d"):
        path = klines_1m_path(root, asset, d)
        atomic_write_parquet(part.drop("_d"), path, schema_name="klines_v1", params={"d": d.isoformat()})


def test_check_klines_passes_on_matching_klines_and_fails_on_a_shifted_minute(tmp_path):
    from pipeline.validate_run import Report, check_klines

    d1, d2 = _one_run(tmp_path)
    bars = pl.concat([pl.read_parquet(p) for p in sorted((tmp_path / "silver" / "bars" / "time_1m").glob("BTCUSDT_*.parquet"))]).sort("ts_ms")
    ladders = {"BTCUSDT": {"time_1m": bars}}
    _write_klines_from(tmp_path, bars)
    good = Report()
    check_klines(good, tmp_path, ladders, d1, d2)
    assert [c.severity for c in good.checks] == ["PASS"]
    # A kline file whose closes are wrong by one minute (the old boundary bug) must fail.
    shifted = bars.with_columns(pl.col("close").shift(1).fill_null(1.0))
    _write_klines_from(tmp_path, shifted)
    bad = Report()
    check_klines(bad, tmp_path, ladders, d1, d2)
    assert bad.checks[0].severity == "FAIL"


def test_check_dollar_fails_on_a_sub_threshold_bar(tmp_path):
    from pipeline.validate_run import Report, check_dollar
    from pipeline.validate_run import _trade_days

    d1, d2 = _one_run(tmp_path)
    ladder_dollar = pl.concat([pl.read_parquet(p) for p in sorted((tmp_path / "silver" / "bars" / "dollar").glob("BTCUSDT_*.parquet"))]).sort("ts_ms")
    days = _trade_days(tmp_path, "BTCUSDT", d1, d2)
    bronze_n = sum(int(pl.scan_parquet(tmp_path / "bronze" / "trades" / "spot" / "BTCUSDT" / f"{d.isoformat()}.parquet").select(pl.len()).collect().item()) for d in days)
    ok = Report()
    check_dollar(ok, tmp_path, "BTCUSDT", days, ladder_dollar, bronze_n, threshold_overrides={"BTCUSDT": 5000.0})
    assert ok.checks[0].severity == "PASS", ok.checks[0].detail
    broken = ladder_dollar.with_columns(
        pl.when(pl.col("bar_id") == 3).then(pl.col("volume_usd") * 0.5).otherwise(pl.col("volume_usd")).alias("volume_usd"),
    )
    bad = Report()
    check_dollar(bad, tmp_path, "BTCUSDT", days, broken, bronze_n, threshold_overrides={"BTCUSDT": 5000.0})
    assert bad.checks[0].severity == "FAIL"


def test_check_manifests_flags_a_tampered_gold_file(tmp_path):
    from pipeline.validate_run import Report, check_manifests

    d1, d2 = _one_run(tmp_path)
    ok = Report()
    check_manifests(ok, tmp_path, d1, d2)
    assert ok.checks[0].severity == "PASS", ok.checks[0].detail
    target = tmp_path / "gold" / "time_1h" / "BTCUSDT_2024-06.parquet"
    target.write_bytes(target.read_bytes() + b"x")
    bad = Report()
    check_manifests(bad, tmp_path, d1, d2)
    assert bad.checks[0].severity == "FAIL"
