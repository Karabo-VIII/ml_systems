"""Validation gates from the data contract.

Exit codes, matching the contract: 0 pass, 1 warn, 2 hard fail.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from pipeline.constants import (
    BTC_MAX_GAP_MS,
    CONSTANT_ALLOW,
    MARKET_COLUMNS,
    TIME_WIDTH_MS,
)
from pipeline.features import PHYSICS_FEATURES

# Every physics feature is built from the bars alone (the funding z-scores were
# dropped in 2.5.0), so an all-null or constant one is a defect.
OHLCV_FEATURES = tuple(PHYSICS_FEATURES)
# Features whose lookahead failure fails the run. FX features only warn: 1-minute
# FX mid returns have real short-lag autocorrelation, so a trailing return can
# exceed the threshold without seeing the future.
LOOKAHEAD_CORE = tuple(OHLCV_FEATURES)


@dataclass
class GateResult:
    name: str
    severity: str  # pass, warn, fail
    detail: str

    @property
    def code(self) -> int:
        return {"pass": 0, "warn": 1, "fail": 2}[self.severity]


@dataclass
class GateReport:
    results: list[GateResult] = field(default_factory=list)

    def add(self, name: str, severity: str, detail: str) -> None:
        self.results.append(GateResult(name, severity, detail))

    @property
    def code(self) -> int:
        if any(r.severity == "fail" for r in self.results):
            return 2
        if any(r.severity == "warn" for r in self.results):
            return 1
        return 0

    def failures(self) -> list[GateResult]:
        return [r for r in self.results if r.severity == "fail"]


def _dtype_eq(actual: pl.DataType, expected: pl.DataType) -> bool:
    a, b = str(actual), str(expected)
    if a == b:
        return True
    return {a, b} <= {"String", "Utf8"}


def _rel_close(a: float, b: float, rel: float = 1e-9, abs_tol: float = 1e-8) -> bool:
    return abs(a - b) <= abs_tol + rel * max(abs(a), abs(b))


def gate_schema(df: pl.DataFrame, required: list[str], dtypes: dict[str, pl.DataType] | None = None) -> GateResult:
    missing = [c for c in required if c not in df.columns]
    if missing:
        return GateResult("schema", "fail", f"missing columns {missing}")
    if dtypes:
        bad = []
        for col, dt in dtypes.items():
            if col not in df.columns:
                continue
            if not _dtype_eq(df.schema[col], dt):
                bad.append(f"{col}:{df.schema[col]}!={dt}")
        if bad:
            return GateResult("schema", "fail", "dtype mismatch " + ", ".join(bad[:8]))
    return GateResult("schema", "pass", f"{len(required)} required columns present")


def gate_monotonic_unique(df: pl.DataFrame, col: str = "ts_ms") -> GateResult:
    if df.is_empty():
        return GateResult("monotonic", "warn", "empty frame")
    values = df[col].to_list()
    if any(b < a for a, b in zip(values, values[1:])):
        return GateResult("monotonic", "fail", f"{col} is not monotone")
    if df[col].n_unique() != df.height:
        return GateResult("monotonic", "fail", f"{col} is not unique ({df[col].n_unique()} vs {df.height})")
    return GateResult("monotonic", "pass", f"{col} strict-monotone, n={df.height}")


def gate_ohlc(df: pl.DataFrame) -> GateResult:
    if df.is_empty():
        return GateResult("ohlc", "warn", "empty")
    bad_hl = df.filter(pl.col("high") < pl.col("low")).height
    bad_h = df.filter((pl.col("high") + 1e-9 < pl.col("open")) | (pl.col("high") + 1e-9 < pl.col("close"))).height
    bad_l = df.filter((pl.col("low") - 1e-9 > pl.col("open")) | (pl.col("low") - 1e-9 > pl.col("close"))).height
    neg = df.filter((pl.col("volume") < 0) | (pl.col("volume_usd") < 0)).height
    if bad_hl or bad_h or bad_l or neg:
        return GateResult("ohlc", "fail", f"hl={bad_hl} high={bad_h} low={bad_l} negvol={neg}")
    # Spot FX has no aggressor flag. A null side is not a broken partition.
    sided = df.filter(pl.col("buy_volume").is_not_null() & pl.col("sell_volume").is_not_null())
    if sided.is_empty():
        return GateResult("ohlc", "pass", "OHLC holds; no aggressor side on this market")
    # buy + sell partitions volume
    gap = sided.select(
        (pl.col("buy_volume") + pl.col("sell_volume") - pl.col("volume")).abs().alias("dv"),
        pl.col("volume").abs().alias("v"),
    )
    worst = 0.0
    for dv, v in zip(gap["dv"].to_list(), gap["v"].to_list()):
        tol = 1e-8 + 1e-9 * max(v, 1e-12)
        if dv > tol:
            worst = max(worst, dv)
    if worst > 0:
        return GateResult("ohlc", "fail", f"buy+sell diverges from volume by {worst}")
    return GateResult("ohlc", "pass", "OHLC and buy/sell partition hold")


def gate_time_reconcile(ladder: dict[str, pl.DataFrame]) -> GateResult:
    """Sum of 1m volume equals each coarser bar's volume. Same for n_trades and volume_usd."""
    if "time_1m" not in ladder or ladder["time_1m"].is_empty():
        return GateResult("reconcile", "warn", "no 1m bars")
    base = ladder["time_1m"]
    problems = []
    for name, width in TIME_WIDTH_MS.items():
        if name == "time_1m" or name not in ladder:
            continue
        coarse = ladder[name]
        if coarse.is_empty():
            problems.append(f"{name} empty")
            continue
        for col in ("volume", "volume_usd", "n_trades"):
            if base[col].drop_nulls().len() == 0 and coarse[col].drop_nulls().len() == 0:
                continue
            b = float(base[col].sum())
            c = float(coarse[col].sum())
            if col == "n_trades":
                if int(b) != int(c):
                    problems.append(f"{name}.{col} {b} != {c}")
            elif not _rel_close(b, c):
                problems.append(f"{name}.{col} {b} != {c}")
    if problems:
        return GateResult("reconcile", "fail", "; ".join(problems[:6]))
    return GateResult("reconcile", "pass", "1m sums match coarser time bars")


def gate_trade_count(n_trades_in_bars: int, n_bronze: int, *, excluded_tail: int = 0, what: str = "bars") -> GateResult:
    if n_trades_in_bars + excluded_tail != n_bronze:
        return GateResult(
            "trade_count", "fail",
            f"{what}: bars {n_trades_in_bars} + tail {excluded_tail} != bronze {n_bronze}",
        )
    return GateResult("trade_count", "pass", f"{what}: {n_bronze} trades accounted for")


def gate_gaps(missing_ms: list[int], *, asset: str, max_hole_ms: int = BTC_MAX_GAP_MS) -> GateResult:
    from pipeline.bars import contiguous_hole_ms
    hole = contiguous_hole_ms(missing_ms)
    if asset == "BTCUSDT" and hole > max_hole_ms:
        return GateResult("gaps", "fail", f"BTC contiguous hole {hole} ms exceeds {max_hole_ms}")
    return GateResult("gaps", "pass", f"gaps={len(missing_ms)} longest_hole_ms={hole}")


def gate_constants(
    df: pl.DataFrame,
    cols: list[str],
    *,
    min_rows: int = 50,
    min_history: int = 50,
) -> list[GateResult]:
    """A non-indicator column with a single distinct value is an invented constant.

    An all-null column on a frame shorter than the feature lookback is warmup,
    not a dead series. On a long frame, an all-null core feature fails.
    """
    results = []
    short = df.height < min_history
    if "is_warmup" in df.columns:
        # A short-lived asset can have more rows than min_history yet almost none
        # past its warmup: a slow feature that never settled is still warmup.
        settled = df.height - int(df["is_warmup"].fill_null(True).sum())
        short = short or settled < min_rows
    for col in cols:
        if col not in df.columns or col in CONSTANT_ALLOW or col.startswith("target_"):
            continue
        s = df[col].drop_nulls()
        if s.len() == 0:
            if short:
                continue
            if col in OHLCV_FEATURES:
                results.append(GateResult("constant", "fail", f"{col} is entirely null"))
            continue
        if s.len() >= min_rows and s.n_unique() == 1:
            level = "fail" if col in OHLCV_FEATURES else "warn"
            results.append(GateResult("constant", level, f"{col} is constant ({s[0]}) over {s.len()} rows"))
    if not results:
        results.append(GateResult("constant", "pass", "no invented constants"))
    return results


def _n_pairs(x: np.ndarray, y: np.ndarray) -> int:
    return int((np.isfinite(x) & np.isfinite(y)).sum())


def _finite_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 30:
        return None
    return x[mask], y[mask]


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    pair = _finite_pair(x, y)
    if pair is None:
        return None
    a, b = pair
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if denom < 1e-15:
        return None
    return float(np.dot(a, b) / denom)


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    pair = _finite_pair(x, y)
    if pair is None:
        return None
    a, b = pair

    def rank(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="mergesort")
        out = np.empty(len(v), dtype=np.float64)
        out[order] = np.arange(len(v))
        return out

    return _pearson(rank(a), rank(b))


def gate_lookahead(
    df: pl.DataFrame,
    feature_cols: list[str],
    *,
    threshold: float = 0.20,
    target: str = "target_return_1",
    fail_z: float = 5.0,
) -> list[GateResult]:
    """Correlation of each feature with the forward return: a smoke alarm.

    A core feature (crypto physics) fails the run when its |corr| is over the
    threshold AND significant: |corr| * sqrt(n) >= fail_z, i.e. at least five
    standard errors under independence. On a short frame (a coin listed late
    in the year, a 1d cadence) a 0.2 correlation is noise and only warns; the
    causality proof (truncation recompute in the validate stage) is what proves
    a feature uses no future data. Any other feature, FX included, warns.
    Nothing is removed from gold: the run's exit code stops a leaking build.
    """
    if target not in df.columns:
        return [GateResult("lookahead", "fail", f"missing {target}")]
    y = df[target].to_numpy().astype(np.float64)
    results: list[GateResult] = []
    for col in feature_cols:
        if col not in df.columns or col.startswith("target_"):
            continue
        x = df[col].to_numpy().astype(np.float64)
        pear = _pearson(x, y)
        spear = _spearman(x, y)
        if pear is None and spear is None:
            continue
        worst = max(abs(pear or 0.0), abs(spear or 0.0))
        if worst > threshold:
            n = _n_pairs(x, y)
            z = worst * float(np.sqrt(n))
            level = "fail" if col in LOOKAHEAD_CORE and z >= fail_z else "warn"
            results.append(GateResult(
                "lookahead", level,
                f"{col} |corr|={worst:.3f} > {threshold} vs {target} on n={n} "
                f"(z={z:.1f}, fail needs {fail_z:g}; possible lookahead)",
            ))
    if not any(r.severity == "fail" for r in results):
        if not results:
            results.append(GateResult("lookahead", "pass", f"no feature exceeded {threshold}"))
        else:
            results.insert(0, GateResult("lookahead", "pass", "no core feature exceeded the threshold"))
    return results


def gate_asof_control(base: pl.DataFrame, feature: pl.DataFrame, lag_ms: int, value_col: str) -> GateResult:
    """A value with lag L must not appear on a bar with ts_ms < event_time + L."""
    from pipeline.align import asof_join_knowable
    if base.is_empty() or feature.is_empty():
        return GateResult("asof", "warn", "empty as-of control")
    joined = asof_join_knowable(base, feature, [value_col], lag_ms)
    feat = feature.select(["ts_ms", value_col]).rename({"ts_ms": "_event"})
    # For every attached value, find the event time and check knowable <= bar ts.
    check = joined.select(["ts_ms", value_col]).join(feat, on=value_col, how="left")
    # A value may match several events; the join above is not the as-of. Re-check directly:
    events = feature.select([
        pl.col("ts_ms").alias("event_ms"),
        (pl.col("ts_ms") + lag_ms).alias("knowable"),
        pl.col(value_col),
    ]).sort("knowable")
    bad = 0
    values = joined[value_col].to_list()
    times = joined["ts_ms"].to_list()
    ev_know = list(zip(events["knowable"].to_list(), events[value_col].to_list()))
    for ts, val in zip(times, values):
        if val is None:
            continue
        # The attached value must have some event with the same value whose knowable <= ts,
        # and there must not be a strictly later knowable that was the one selected... 
        # Stronger: the selected event is the last with knowable <= ts, and that knowable <= ts.
        candidates = [k for k, v in ev_know if v == val and k <= ts]
        if not candidates:
            bad += 1
    if bad:
        return GateResult("asof", "fail", f"{bad} rows have {value_col} before it was knowable")
    return GateResult("asof", "pass", f"lag {lag_ms} ms respected for {value_col}")


def gate_splits(df: pl.DataFrame, embargo_bars: int) -> GateResult:
    if "split" not in df.columns or df.is_empty():
        return GateResult("splits", "warn", "no split column")
    allowed = {"train", "val", "oos", "unseen", "embargo"}
    unknown = set(df["split"].drop_nulls().unique().to_list()) - allowed
    if unknown:
        return GateResult("splits", "fail", f"unknown split labels {unknown}")
    # No row in two segments: one column, so this is structural. Check embargo width
    # where both neighbouring segments are at least embargo_bars long.
    labels = df.sort("ts_ms")["split"].to_list()
    # Walk runs.
    runs: list[tuple[str, int]] = []
    for lab in labels:
        if not runs or runs[-1][0] != lab:
            runs.append((str(lab), 1))
        else:
            runs[-1] = (runs[-1][0], runs[-1][1] + 1)
    order = ["train", "val", "oos", "unseen"]
    for i in range(len(runs) - 2):
        a, an = runs[i]
        mid, mn = runs[i + 1]
        b, bn = runs[i + 2]
        if mid != "embargo":
            continue
        if a not in order or b not in order or order.index(b) <= order.index(a):
            continue
        if an >= embargo_bars and bn >= embargo_bars and mn < 2 * embargo_bars:
            return GateResult(
                "splits", "fail",
                f"embargo between {a} and {b} is {mn} bars, need {2 * embargo_bars}",
            )
    return GateResult("splits", "pass", f"embargo request {embargo_bars} bars")


def gate_cross_asset_nulls(df: pl.DataFrame, n_assets_in_run: int) -> GateResult:
    xd = [c for c in df.columns if c.startswith("xd_")]
    if n_assets_in_run < 2:
        for c in xd:
            if df[c].drop_nulls().len() > 0:
                return GateResult("universe", "fail", f"{c} is non-null on a single-asset run")
        return GateResult("universe", "pass", "cross-asset features null on a single-asset run")
    return GateResult("universe", "pass", f"cross-asset checked with {n_assets_in_run} assets")


def market_dtype_map() -> dict[str, pl.DataType]:
    return {
        "asset": pl.Utf8,
        "bar_type": pl.Utf8,
        "bar_id": pl.Int64,
        "ts_open_ms": pl.Int64,
        "ts_ms": pl.Int64,
        "n_trades": pl.Int64,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
        "volume_usd": pl.Float64,
    }


def validate_frame(
    df: pl.DataFrame,
    *,
    feature_cols: list[str],
    embargo_bars: int,
    n_assets_in_run: int,
    asset: str,
    missing_minutes: list[int] | None = None,
    min_history: int = 50,
) -> GateReport:
    report = GateReport()
    required = [
        "asset", "bar_type", "bar_id", "ts_open_ms", "ts_ms",
        "open", "high", "low", "close", "volume", "volume_usd", "n_trades",
    ]
    schema = gate_schema(df, required, market_dtype_map())
    report.results.append(schema)
    report.results.append(gate_monotonic_unique(df, "ts_ms"))
    if schema.severity != "fail":
        report.results.append(gate_ohlc(df))
    for item in gate_constants(df, feature_cols, min_history=min_history):
        report.results.append(item)
    for item in gate_lookahead(df, feature_cols):
        report.results.append(item)
    report.results.append(gate_splits(df, embargo_bars))
    report.results.append(gate_cross_asset_nulls(df, n_assets_in_run))
    if missing_minutes is not None:
        report.results.append(gate_gaps(missing_minutes, asset=asset))
    return report
