"""Causal joins, physics, labels, embargo, lookahead."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import polars as pl

from pipeline.align import asof_join_knowable
from pipeline.bars import trades_to_time
from pipeline.constants import DAY_MS
from pipeline.config import Splits, Windows
from pipeline.dataset import add_targets, apply_embargo, assign_split
from pipeline.features import add_cross_asset
from pipeline.features import compute_physics
from fixtures import make_trades, small_windows
from pipeline.validate import gate_asof_control, gate_lookahead


def _bars(n=400, seed=7):
    start = datetime(2024, 6, 1, 0, 0, 30, tzinfo=timezone.utc)
    trades = make_trades(n, start=start, step=timedelta(minutes=1), price0=100.0, qty=1.0, seed=seed, asset_noise=0.002)
    bars = trades_to_time(trades, 60_000, "BTCUSDT", "time_1m")
    # Varying funding so the funding feature is not a constant.
    funding = pl.DataFrame({
        "ts_ms": [int(bars["ts_ms"][i]) for i in range(0, n, 40)],
        "funding_rate": [0.0001 * ((i % 5) - 2) for i in range(0, n, 40)],
    })
    return asof_join_knowable(bars, funding, ["funding_rate"], 0), trades


def test_asof_does_not_attach_a_value_early():
    base = pl.DataFrame({"ts_ms": [1_000, 2_000, 5_000, 10_000], "asset": ["BTCUSDT"] * 4})
    feat = pl.DataFrame({"ts_ms": [1_000, 4_000], "x": [10.0, 40.0]})
    joined = asof_join_knowable(base, feat, ["x"], publication_lag_ms=3_000)
    # knowable at 4000 and 7000. Bar 1000 and 2000 see nothing. Bar 5000 sees 10. Bar 10000 sees 40.
    assert joined["x"].to_list() == [None, None, 10.0, 40.0]
    assert gate_asof_control(base, feat, 3_000, "x").severity == "pass"


def test_daily_value_is_not_visible_on_the_same_day():
    day = date(2024, 6, 1)
    midnight = int(datetime(2024, 6, 1, tzinfo=timezone.utc).timestamp() * 1000)
    next_midnight = midnight + DAY_MS
    # Intraday bar and the bar that closes exactly at the next midnight.
    base = pl.DataFrame({
        "ts_ms": [midnight + 3_600_000, next_midnight],
        "asset": ["BTCUSDT", "BTCUSDT"],
    })
    panel = pl.DataFrame({"event_time_ms": [midnight], "wh_large_usd": [123.0]})
    joined = asof_join_knowable(base, panel, ["wh_large_usd"], DAY_MS, feature_time_col="event_time_ms")
    assert joined["wh_large_usd"].to_list() == [None, 123.0]


def test_physics_is_causal_and_does_not_zero_fill_returns():
    bars, _ = _bars(300)
    w = small_windows()
    out = compute_physics(bars, w)
    assert "norm_oi_change" not in out.columns
    assert "norm_bar_duration" not in out.columns  # time bars
    assert out["is_warmup"][: w.max_lookback].all()
    # First return feature is null, not 0, while the window is warming up.
    assert out["norm_return_1"].null_count() > 0
    # Perturb the last close. Earlier feature rows must not move.
    bumped = bars.with_columns(
        pl.when(pl.col("ts_ms") == bars["ts_ms"][-1]).then(pl.col("close") * 1.5).otherwise(pl.col("close")).alias("close")
    )
    # Keep OHLC sane after the bump.
    bumped = bumped.with_columns([
        pl.max_horizontal("open", "close", "high").alias("high"),
        pl.min_horizontal("open", "close", "low").alias("low"),
    ])
    out2 = compute_physics(bumped, w)
    check = [c for c in out.columns if c.startswith("norm_") or c == "hurst_regime"]
    for col in check:
        a = out[col].to_numpy()[:-1]
        b = out2[col].to_numpy()[:-1]
        both = ~(np.isnan(a.astype(float)) | np.isnan(b.astype(float)))
        # Nulls must stay null together, and finite values must match.
        assert np.array_equal(np.isnan(a.astype(float)), np.isnan(b.astype(float))), col
        if both.any():
            assert np.allclose(a.astype(float)[both], b.astype(float)[both], equal_nan=True), col


def test_include_oi_is_opt_in_and_lagged():
    bars, _ = _bars(80)
    w = small_windows(z=10, frac_diff=15, hurst=10, ma=10, perm_entropy=10, kurtosis=8, kyle=8, yz=8, cs_block=4, vol_cluster=8)
    # An OI print at the same instant as a bar must not attach when the lag is 5 minutes.
    oi_ts = int(bars["ts_ms"][10])
    metrics = pl.DataFrame({"ts_ms": [oi_ts], "oi_value": [1_000_000.0]})
    from pipeline.config import lag_ms
    OI_LAG_MS = lag_ms("open_interest")
    from pipeline.align import asof_join_knowable as join
    lagged = join(bars, metrics, ["oi_value"], OI_LAG_MS)
    assert lagged["oi_value"][10] is None
    later = lagged.filter(pl.col("ts_ms") >= oi_ts + OI_LAG_MS)
    assert later["oi_value"].drop_nulls().len() > 0
    featured = compute_physics(lagged, w, include_oi=True)
    assert "norm_oi_change" in featured.columns


def test_dollar_duration_exists_and_time_bars_skip_it():
    w = small_windows(z=8, frac_diff=10, hurst=8, ma=8, perm_entropy=8, kurtosis=6, kyle=6, yz=6, cs_block=3, vol_cluster=6)
    start = datetime(2024, 6, 1, tzinfo=timezone.utc)
    trades = make_trades(40, start=start, step=timedelta(seconds=30), price0=10, qty=1, seed=9, asset_noise=0.01)
    from pipeline.bars import dollar_bars
    bars, _ = dollar_bars(trades, threshold=15.0, asset="BTCUSDT")
    if bars.height < 15:
        bars, _ = dollar_bars(trades, threshold=5.0, asset="BTCUSDT")
    out = compute_physics(bars, w)
    assert "norm_bar_duration" in out.columns


def test_targets_are_uncapped_and_null_at_the_tail():
    bars, _ = _bars(80)
    out = add_targets(bars, (1, 4), clipped=False, voladj=False)
    assert "target_voladj_1" not in out.columns
    assert "target_return_1_clipped" not in out.columns
    h = 4
    manual = (out["close"].shift(-h) - out["close"]) / out["close"]
    assert np.allclose(
        out["target_return_4"].to_numpy()[:-h].astype(float),
        manual.to_numpy()[:-h].astype(float),
    )
    assert out["target_return_4"].tail(h).null_count() == h
    # A move larger than the old 15% clip is preserved.
    wide = bars.with_columns(pl.when(pl.arange(0, pl.len()) == 0).then(pl.col("close")).otherwise(pl.col("close") * 2).alias("close"))
    # simpler: just check the formula is not clipped by constructing two closes
    tiny = pl.DataFrame({"close": [100.0, 130.0], "ts_ms": [1, 2]})
    labelled = add_targets(tiny, (1,))
    assert labelled["target_return_1"][0] == (130.0 - 100.0) / 100.0


def test_embargo_is_not_shrunk():
    # 30 hourly bars straddling 2023-07-01.
    start = int(datetime(2023, 6, 30, tzinfo=timezone.utc).timestamp() * 1000)
    ts = [start + i * 3_600_000 for i in range(40)]
    df = pl.DataFrame({"ts_ms": ts, "close": [1.0] * 40})
    splits = Splits(
        train_end=date(2023, 7, 1),
        val_end=date(2024, 5, 15),
        oos_end=date(2025, 3, 15),
        unseen_start=date(2025, 3, 15),
        live_start=date(2025, 9, 3),
    )
    df = assign_split(df, splits)
    df = apply_embargo(df, embargo_bars=3)
    labels = df["split"].to_list()
    assert "embargo" in labels
    # The boundary neighbourhood is exactly 3 on each side, not a 5% cap.
    assert labels.count("embargo") == 6


def test_single_asset_cross_section_is_null_not_zero():
    bars, _ = _bars(60)
    w = small_windows(z=8, frac_diff=10, hurst=8, ma=8, perm_entropy=8, kurtosis=6, kyle=6, yz=6, cs_block=3, vol_cluster=6, fast=5)
    feat = compute_physics(bars, w)
    out = add_cross_asset({"BTCUSDT": feat}, bar_type="time_1m", fast=5, ma=8)
    frame = out["BTCUSDT"]
    for col in ("xd_btc_return", "xd_cross_return_mean", "xd_momentum_rank"):
        assert col in frame.columns
        assert frame[col].drop_nulls().len() == 0


def test_two_assets_get_a_peer_return():
    start = datetime(2024, 6, 1, 0, 0, 30, tzinfo=timezone.utc)
    btc = trades_to_time(make_trades(80, start=start, step=timedelta(minutes=1), price0=100, seed=1, asset_noise=0.003), 60_000, "BTCUSDT", "time_1m")
    eth = trades_to_time(make_trades(80, start=start, step=timedelta(minutes=1), price0=10, seed=2, asset_noise=0.003), 60_000, "ETHUSDT", "time_1m")
    out = add_cross_asset({"BTCUSDT": btc, "ETHUSDT": eth}, bar_type="time_1m", fast=5, ma=8)
    eth_x = out["ETHUSDT"]["xd_btc_return"].drop_nulls()
    assert eth_x.len() > 0
    # Rank needs 3 names, so it stays null with only two.
    assert out["ETHUSDT"]["xd_momentum_rank"].drop_nulls().len() == 0


def test_lookahead_gate_catches_a_planted_leak_and_passes_lagged_returns():
    bars, _ = _bars(400)
    w = small_windows()
    feat = compute_physics(bars, w)
    labelled = add_targets(feat, (1, 4))
    cols = [c for c in labelled.columns if c.startswith("norm_") or c == "hurst_regime"]
    report = gate_lookahead(labelled, cols)
    assert all(r.severity != "fail" for r in report), [r.detail for r in report if r.severity == "fail"]
    leaked = labelled.with_columns(pl.col("target_return_1").alias("norm_leak"))
    # Pretend the leak is a core OHLCV feature by using a name the gate treats as core.
    # norm_return_1 is core; overwrite it with the future return.
    leaked = labelled.with_columns(pl.col("target_return_1").alias("norm_return_1"))
    bad = gate_lookahead(leaked, ["norm_return_1"])
    assert any(r.severity == "fail" for r in bad)
