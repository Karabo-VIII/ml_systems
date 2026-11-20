#!/usr/bin/env python3
"""Independent checks on a built pipeline data root.

Run from the repository root (PowerShell or bash):

    python scripts/validate_run.py --data-root data

Exit codes match the contract: 0 pass, 1 warn, 2 fail.
Network cross-checks can be skipped with --skip-network.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pipeline.bars import close_of, gaps_ms
from pipeline.constants import DAY_MS, TIME_WIDTH_MS
from pipeline.config import dollar_threshold, fx_pip, whale_large_usd
from pipeline.fetch_crypto import http_get, parse_klines
from pipeline.features import compute_fx_features, summarize_day, rows_to_frame
from pipeline.paths import fx_hour_path, klines_1m_path, trades_path
from pipeline.symbols import canonical_instrument


SEVERITY_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}
FX_PAIRS = {"EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"}
# Contract: FX has no dollar volume or aggressor sides. usd_strength_ret is
# filled only when several USD pairs are present.
FX_NULL_OK = {
    "volume_usd", "buy_volume", "sell_volume", "buy_usd", "sell_usd", "signed_usd",
    "usd_strength_ret",
}
SESSION_ZONES = {
    "is_tokyo": ("Asia/Tokyo", 9, 18),
    "is_london": ("Europe/London", 8, 17),
    "is_newyork": ("America/New_York", 8, 17),
}


@dataclass
class Check:
    name: str
    severity: str
    detail: str


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, severity: str, detail: str) -> None:
        self.checks.append(Check(name, severity, detail))

    @property
    def code(self) -> int:
        if any(c.severity == "FAIL" for c in self.checks):
            return 2
        if any(c.severity == "WARN" for c in self.checks):
            return 1
        return 0

    def print(self) -> None:
        counts = defaultdict(int)
        for c in self.checks:
            counts[c.severity] += 1
        width = max((len(c.name) for c in self.checks), default=8)
        for c in self.checks:
            print(f"  [{c.severity:<4}] {c.name:<{width}}  {c.detail}")
        print()
        print(
            f"SUMMARY  PASS={counts['PASS']}  WARN={counts['WARN']}  "
            f"FAIL={counts['FAIL']}  EXIT={self.code}"
        )


def _is_fx(asset: str) -> bool:
    return canonical_instrument(asset) in FX_PAIRS or (
        len(canonical_instrument(asset)) == 6 and not canonical_instrument(asset).endswith("USDT")
    )


def _latest(directory: Path, asset: str) -> Path | None:
    if not directory.exists():
        return None
    files = sorted(p for p in directory.glob(f"{asset}_*.parquet") if p.is_file() and ".tmp." not in p.name)
    return files[-1] if files else None


def _load_asset_frames(directory: Path, asset: str) -> pl.DataFrame | None:
    """Concat every monthly (or legacy) partition for one asset."""
    from pipeline.partitions import load_partitions
    return load_partitions(directory, asset)


def _frame_for(path: Path) -> pl.DataFrame:
    asset = path.name.rsplit("_", 1)[0]
    loaded = _load_asset_frames(path.parent, asset)
    return loaded if loaded is not None else pl.read_parquet(path)


def discover(root: Path) -> dict[str, dict[str, dict[str, Path]]]:
    """layer -> bar_type -> asset -> latest path (used to find the set of assets).

    Frames are loaded by concatenating every ``ASSET_*.parquet`` partition.
    """
    found: dict[str, dict[str, dict[str, Path]]] = {"gold": {}, "silver": {}}
    gold = root / "gold"
    if gold.exists():
        for folder in sorted(p for p in gold.iterdir() if p.is_dir()):
            by_asset: dict[str, Path] = {}
            for path in folder.glob("*.parquet"):
                if ".tmp." in path.name:
                    continue
                asset = path.name.rsplit("_", 1)[0]
                prev = by_asset.get(asset)
                if prev is None or path.name > prev.name:
                    by_asset[asset] = path
            if by_asset:
                found["gold"][folder.name] = by_asset
    silver = root / "silver" / "bars"
    if silver.exists():
        for folder in sorted(p for p in silver.iterdir() if p.is_dir() and not p.name.startswith("_")):
            by_asset: dict[str, Path] = {}
            for path in folder.glob("*.parquet"):
                if ".tmp." in path.name:
                    continue
                asset = path.name.rsplit("_", 1)[0]
                prev = by_asset.get(asset)
                if prev is None or path.name > prev.name:
                    by_asset[asset] = path
            if by_asset:
                found["silver"][folder.name] = by_asset
    return found


def _utc(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)


def _rel_close(a: float, b: float, rel: float = 1e-9, abs_tol: float = 1e-6) -> bool:
    return abs(a - b) <= abs_tol + rel * max(abs(a), abs(b))


def _fx_weekend_expected(ts_ms: int) -> bool:
    """True when a missing 1m close sits in the usual Friday-Sunday FX hole.

    Close is Friday 17:00 America/New_York; open is Sunday 17:00 America/New_York.
    Those instants move with US daylight-saving time.
    """
    dt = _utc(ts_ms)
    ny = dt.astimezone(ZoneInfo("America/New_York"))
    # The bar *closing* at ts_ms. A gap close during the weekend shutdown is expected.
    if ny.weekday() == 5:
        return True
    if ny.weekday() == 6 and (ny.hour, ny.minute) < (17, 0):
        return True
    if ny.weekday() == 4 and (ny.hour, ny.minute) >= (17, 0):
        return True
    return False


def _longest_run_minutes(ts_list: list[int]) -> int:
    """Longest consecutive 1-minute run in a sorted list of missing closes."""
    if not ts_list:
        return 0
    longest = run = 1
    for a, b in zip(ts_list, ts_list[1:]):
        if b - a == 60_000:
            run += 1
            longest = max(longest, run)
        else:
            run = 1
    return longest


def _lookback_bars(col: str) -> int | None:
    m = re.search(r"_(\d+)$", col)
    return int(m.group(1)) if m else None


def _session_overlap(open_ms: int, close_ms: int, zone: str, start_h: int, end_h: int) -> bool:
    """Bar (open, close] overlaps local [start_h, end_h) on any nearby local day."""
    tz = ZoneInfo(zone)
    start_day = _utc(open_ms).astimezone(tz).date()
    for delta in (-1, 0, 1):
        day = start_day + timedelta(days=delta)
        s = datetime(day.year, day.month, day.day, start_h, 0, tzinfo=tz)
        e = datetime(day.year, day.month, day.day, end_h, 0, tzinfo=tz)
        s_ms, e_ms = int(s.timestamp() * 1000), int(e.timestamp() * 1000)
        if open_ms < e_ms and close_ms > s_ms:
            return True
    return False


def check_monotonic(report: Report, frames: dict[tuple[str, str], pl.DataFrame]) -> None:
    for (asset, bt), df in sorted(frames.items()):
        name = f"monotonic:{asset}:{bt}"
        if df.is_empty():
            report.add(name, "WARN", "empty frame")
            continue
        ts = df["ts_ms"]
        n_unique = ts.n_unique()
        diffs = ts.diff().drop_nulls()
        if n_unique != df.height:
            report.add(name, "FAIL", f"duplicate ts_ms: unique={n_unique} rows={df.height}")
        elif diffs.len() and int((diffs <= 0).sum()) > 0:
            report.add(name, "FAIL", "ts_ms is not strictly increasing")
        else:
            report.add(name, "PASS", f"strict-monotone unique ts_ms n={df.height}")


def check_time_reconcile(
    report: Report, ladders: dict[str, dict[str, pl.DataFrame]],
) -> None:
    for asset, ladder in sorted(ladders.items()):
        base = ladder.get("time_1m")
        if base is None or base.is_empty():
            report.add(f"reconcile:{asset}", "WARN", "no 1m bars")
            continue
        problems = []
        for name, width in TIME_WIDTH_MS.items():
            if name == "time_1m" or name not in ladder:
                continue
            coarse = ladder[name]
            if coarse.is_empty():
                problems.append(f"{name} empty")
                continue
            # Clock-bucket OHLC and sums from 1m children.
            tagged = base.sort("ts_ms").with_columns(
                close_of(pl.col("ts_ms"), width).alias("_close")
            )
            expected = tagged.group_by("_close", maintain_order=True).agg([
                pl.col("open").sort_by("ts_open_ms").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").sort_by("ts_ms").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
                pl.col("n_trades").sum().cast(pl.Int64).alias("n_trades"),
            ])
            joined = coarse.select(["ts_ms", "ts_open_ms", "open", "high", "low", "close", "volume", "n_trades"]).join(
                expected, left_on="ts_ms", right_on="_close", how="left", suffix="_e"
            )
            if joined["open_e"].null_count():
                problems.append(f"{name}: {joined['open_e'].null_count()} coarse bars have no 1m children")
            bad_open = joined.filter(pl.col("ts_open_ms") != (pl.col("ts_ms") - width)).height
            if bad_open:
                problems.append(f"{name}: {bad_open} bars have ts_open_ms != ts_ms - width")
            for col in ("open", "high", "low", "close"):
                mismatch = joined.filter(
                    (pl.col(f"{col}_e").is_not_null())
                    & ((pl.col(col) - pl.col(f"{col}_e")).abs() > 1e-10)
                ).height
                if mismatch:
                    problems.append(f"{name}.{col} mismatches on {mismatch} bars")
            for col in ("volume", "n_trades"):
                b, c = float(base[col].sum()), float(coarse[col].sum())
                if col == "n_trades":
                    if int(b) != int(c):
                        problems.append(f"{name}.{col} sum {c} != 1m {b}")
                elif not _rel_close(b, c):
                    problems.append(f"{name}.{col} sum {c} != 1m {b}")
        if problems:
            report.add(f"reconcile:{asset}", "FAIL", "; ".join(problems[:8]))
        else:
            kinds = [k for k in TIME_WIDTH_MS if k in ladder and k != "time_1m"]
            report.add(
                f"reconcile:{asset}",
                "PASS",
                f"1m OHLCV and trade counts match {', '.join(kinds)}",
            )


def _trade_days(root: Path, asset: str) -> list[date]:
    folder = root / "bronze" / "trades" / "spot" / asset
    if not folder.exists():
        return []
    out = []
    for path in sorted(folder.glob("*.parquet")):
        try:
            out.append(date.fromisoformat(path.stem))
        except ValueError:
            continue
    return out


def _fx_tick_days(root: Path, asset: str) -> list[date]:
    folder = root / "bronze" / "fx_ticks" / asset
    if not folder.exists():
        return []
    out = []
    for path in sorted(p for p in folder.iterdir() if p.is_dir()):
        try:
            out.append(date.fromisoformat(path.name))
        except ValueError:
            continue
    return out


def _read_trades(root: Path, asset: str, days: list[date]) -> pl.DataFrame:
    frames = []
    for day in days:
        path = trades_path(root, asset, day)
        if path.exists():
            frames.append(pl.read_parquet(path))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def _read_fx_ticks(root: Path, asset: str, days: list[date]) -> pl.DataFrame:
    frames = []
    for day in days:
        for hour in range(24):
            path = fx_hour_path(root, asset, day, hour)
            if path.exists():
                frame = pl.read_parquet(path)
                if not frame.is_empty():
                    frames.append(frame)
    if not frames:
        return pl.DataFrame(schema={"ts_ms": pl.Int64, "bid": pl.Float64, "ask": pl.Float64, "mid": pl.Float64, "spread": pl.Float64})
    return pl.concat(frames, how="vertical").sort("ts_ms")


def check_trade_counts(
    report: Report,
    root: Path,
    ladders: dict[str, dict[str, pl.DataFrame]],
) -> None:
    for asset, ladder in sorted(ladders.items()):
        bars = ladder.get("time_1m")
        if bars is None or bars.is_empty():
            continue
        bar_n = int(bars["n_trades"].sum())
        if _is_fx(asset):
            days = _fx_tick_days(root, asset)
            ticks = _read_fx_ticks(root, asset, days)
            bronze = ticks.height
            name = f"tick_count:{asset}"
            if bar_n != bronze:
                report.add(name, "FAIL", f"1m n_trades {bar_n} != tick files {bronze}")
            else:
                report.add(name, "PASS", f"{bronze} ticks accounted for in 1m bars")
            if "spread" in ticks.columns and not ticks.is_empty():
                check_spreads(report, asset, ticks, bars)
        else:
            days = _trade_days(root, asset)
            trades = _read_trades(root, asset, days)
            bronze = trades.height
            name = f"trade_count:{asset}"
            if bar_n != bronze:
                report.add(name, "FAIL", f"1m n_trades {bar_n} != bronze trades {bronze}")
            else:
                report.add(name, "PASS", f"{bronze} trades accounted for in 1m bars")
            dollar = ladder.get("dollar")
            if dollar is not None and not trades.is_empty():
                check_dollar(report, asset, trades, dollar)


def check_dollar(report: Report, asset: str, trades: pl.DataFrame, dollar: pl.DataFrame) -> None:
    name = f"dollar_notional:{asset}"
    if dollar.is_empty():
        report.add(name, "WARN", "no dollar bars")
        return
    threshold = float(dollar_threshold(asset))
    usd = (trades["price"] * trades["qty"]).sum()
    bar_usd = float(dollar["volume_usd"].sum())
    bar_n = int(dollar["n_trades"].sum())
    bronze_n = trades.height
    tail_n = bronze_n - bar_n
    if tail_n < 0:
        report.add(name, "FAIL", f"dollar n_trades {bar_n} exceeds bronze {bronze_n}")
        return
    # Tail notional is whatever was not closed into a bar.
    leftover = float(usd) - bar_usd
    if leftover < -1e-2:
        report.add(name, "FAIL", f"dollar volume_usd {bar_usd} exceeds trade notional {float(usd)}")
        return
    bad = dollar.filter(pl.col("volume_usd") <= 0).height
    if bad:
        report.add(name, "FAIL", f"{bad} dollar bars have non-positive notional")
        return
    med = float(dollar["volume_usd"].median())
    mx = float(dollar["volume_usd"].max())
    # Crossing trades can overshoot; the median should still sit near the threshold.
    if med < 0.2 * threshold or med > 5 * threshold:
        report.add(
            name, "WARN",
            f"median bar notional {med:.0f} is far from threshold {threshold:.0f} "
            f"(max={mx:.0f}, tail_trades={tail_n}, leftover_usd={leftover:.0f})",
        )
        return
    report.add(
        name, "PASS",
        f"{dollar.height} bars, threshold={threshold:.0f}, median_usd={med:.0f}, "
        f"max_usd={mx:.0f}, tail_trades={tail_n}, leftover_usd={leftover:.0f}, "
        f"trades {bar_n}+{tail_n}={bronze_n}",
    )


def check_spreads(report: Report, asset: str, ticks: pl.DataFrame, bars: pl.DataFrame) -> None:
    name = f"spread:{asset}"
    pip = fx_pip(asset)
    if ticks.filter(pl.col("spread") <= 0).height:
        n = ticks.filter(pl.col("spread") <= 0).height
        report.add(name, "FAIL", f"{n} ticks have non-positive spread")
        return
    crossed = ticks.filter(pl.col("ask") < pl.col("bid")).height
    if crossed:
        report.add(name, "FAIL", f"{crossed} ticks have ask < bid")
        return
    mid_bad = ticks.filter(
        ((pl.col("bid") + pl.col("ask")) / 2 - pl.col("mid")).abs() > 1e-10
    ).height
    if mid_bad:
        report.add(name, "FAIL", f"{mid_bad} ticks have mid != (bid+ask)/2")
        return
    if "bid_close" in bars.columns:
        inv = bars.filter(
            pl.col("bid_close").is_not_null()
            & pl.col("ask_close").is_not_null()
            & (pl.col("ask_close") < pl.col("bid_close"))
        ).height
        if inv:
            report.add(name, "FAIL", f"{inv} bars have ask_close < bid_close")
            return
    pips = ticks["spread"] / pip
    p50 = float(pips.median())
    p99 = float(pips.quantile(0.99))
    mx = float(pips.max())
    # Majors: a typical spread is a fraction of a pip to a few pips. >50 pips at p99 is odd.
    if p50 <= 0:
        report.add(name, "FAIL", f"median spread is {p50} pips")
        return
    sev = "WARN" if p99 > 50 else "PASS"
    report.add(
        name, sev,
        f"ticks={ticks.height} median={p50:.2f} pips  p99={p99:.2f}  max={mx:.2f}  pip={pip}",
    )


def check_gaps(report: Report, ladders: dict[str, dict[str, pl.DataFrame]]) -> None:
    for asset, ladder in sorted(ladders.items()):
        bars = ladder.get("time_1m")
        if bars is None or bars.is_empty():
            continue
        missing = gaps_ms(bars)
        name = f"gaps:{asset}"
        if _is_fx(asset):
            unexpected = [t for t in missing if not _fx_weekend_expected(t)]
            weekend = len(missing) - len(unexpected)
            if unexpected:
                # Isolated quiet minutes (no ticks) are real but not a fetch hole.
                # Fail only when a weekday hole is long enough to look like missing hours.
                longest = _longest_run_minutes(unexpected)
                sev = "FAIL" if longest > 15 else "WARN"
                sample = ", ".join(_utc(t).isoformat() for t in unexpected[:5])
                report.add(
                    name, sev,
                    f"{len(unexpected)} missing minutes outside the NY-weekend close "
                    f"(weekend_ok={weekend}, longest hole={longest} min); first: {sample}",
                )
            else:
                report.add(
                    name, "PASS",
                    f"{len(missing)} missing 1m closes, all inside the Friday-Sunday "
                    f"NY 17:00 close",
                )
        else:
            if missing:
                longest = _longest_run_minutes(missing)
                sev = "FAIL" if asset == "BTCUSDT" and longest > 15 else "WARN"
                first = _utc(missing[0]).isoformat()
                report.add(
                    name, sev,
                    f"{len(missing)} missing minutes, longest hole={longest} min, first={first}",
                )
            else:
                first, last = _utc(int(bars["ts_ms"].min())), _utc(int(bars["ts_ms"].max()))
                n_days = (last.date() - first.date()).days
                # Close-labelled: first close to last close inclusive.
                expected = int(bars.height)
                report.add(
                    name, "PASS",
                    f"no missing minutes; {expected} 1m bars from {first.isoformat()} to {last.isoformat()}",
                )


def check_sessions(report: Report, gold: dict[str, dict[str, Path]]) -> None:
    # Prefer 1h gold, fall back to 1m.
    source = gold.get("time_1h") or gold.get("time_1m") or {}
    fx_assets = [a for a in source if _is_fx(a)]
    if not fx_assets:
        return
    mismatches = []
    checked = 0
    dst_notes = []
    for asset in fx_assets:
        df = _frame_for(source[asset])
        need = ["ts_open_ms", "ts_ms", "is_tokyo", "is_london", "is_newyork", "is_ldn_ny_overlap"]
        if any(c not in df.columns for c in need):
            report.add(f"sessions:{asset}", "WARN", "session columns missing")
            continue
        # Confirm the week is internally consistent with the zone database.
        sample_ts = int(df["ts_ms"].median())
        ny = _utc(sample_ts).astimezone(ZoneInfo("America/New_York"))
        ldn = _utc(sample_ts).astimezone(ZoneInfo("Europe/London"))
        dst_notes.append(f"{asset} mid-window NY={ny.tzname()} London={ldn.tzname()}")
        for row in df.select(need).iter_rows(named=True):
            o, c = int(row["ts_open_ms"]), int(row["ts_ms"])
            got = {
                "is_tokyo": int(row["is_tokyo"]),
                "is_london": int(row["is_london"]),
                "is_newyork": int(row["is_newyork"]),
            }
            exp = {
                key: int(_session_overlap(o, c, zone, h0, h1))
                for key, (zone, h0, h1) in SESSION_ZONES.items()
            }
            checked += 1
            if got != exp:
                mismatches.append(
                    f"{asset} {_utc(c).isoformat()} got={got} expected={exp}"
                )
            overlap = int(got["is_london"] and got["is_newyork"])
            if overlap != int(row["is_ldn_ny_overlap"]):
                mismatches.append(f"{asset} {_utc(c).isoformat()} overlap flag wrong")
        if mismatches and len(mismatches) > 8:
            break
    if not checked:
        report.add("sessions", "WARN", "no FX session columns to check")
    elif mismatches:
        report.add("sessions", "FAIL", f"{len(mismatches)} mismatches; " + "; ".join(mismatches[:6]))
    else:
        report.add(
            "sessions", "PASS",
            f"{checked} bars match zone-aware Tokyo/London/NY session bounds; " + "; ".join(dst_notes[:2]),
        )


def check_asof_and_hand_features(
    report: Report,
    root: Path,
    gold_files: dict[str, dict[str, Path]],
    ladders: dict[str, dict[str, pl.DataFrame]],
) -> None:
    # Prefer 1m gold so z-scored features have enough warmup; fall back to 1h.
    for bt in ("time_1m", "time_1h"):
        assets = gold_files.get(bt, {})
        if assets:
            break
    else:
        report.add("hand_features", "WARN", "no gold files")
        return

    ret_problems: list[str] = []
    asof_problems: list[str] = []
    asof_ok = 0
    ret_ok = 0

    def _rolling_z(values, window: int) -> list:
        out = [None] * len(values)
        for i in range(window - 1, len(values)):
            chunk = values[i - window + 1:i + 1]
            if any(v is None for v in chunk):
                continue
            mu = sum(chunk) / window
            var = sum((v - mu) ** 2 for v in chunk) / (window - 1) if window > 1 else 0.0
            sd = var ** 0.5
            x = chunk[-1]
            if sd < 1e-12:
                continue
            z = (x - mu) / (sd + 1e-5)
            out[i] = max(-5.0, min(5.0, z))
        return out

    for asset, path in sorted(assets.items()):
        df = _frame_for(path).sort("ts_ms")
        if "close" not in df.columns or df.height < 3:
            continue
        close = df["close"].to_list()
        ts = df["ts_ms"].to_list()

        if "target_return_1" in df.columns:
            series = df["target_return_1"].to_list()
            for i in range(len(close) - 1):
                if not close[i] or close[i] <= 0 or close[i + 1] is None:
                    continue
                expected = close[i + 1] / close[i] - 1.0
                got = series[i]
                if got is None:
                    ret_problems.append(f"{asset} target_return_1 null at {_utc(ts[i]).isoformat()}")
                elif abs(got - expected) > 1e-10 * max(1.0, abs(expected)):
                    ret_problems.append(
                        f"{asset} target_return_1 {got} != {expected} at {_utc(ts[i]).isoformat()}"
                    )
                    if len(ret_problems) >= 6:
                        break
                else:
                    ret_ok += 1

        if "ret_1" in df.columns:
            series = df["ret_1"].to_list()
            for i in range(1, len(close)):
                if close[i - 1] and close[i - 1] > 0 and close[i] is not None:
                    expected = close[i] / close[i - 1] - 1.0
                    got = series[i]
                    if got is None:
                        ret_problems.append(f"{asset} ret_1 null at {_utc(ts[i]).isoformat()}")
                    elif abs(got - expected) > 1e-10 * max(1.0, abs(expected)):
                        ret_problems.append(
                            f"{asset} ret_1 {got} != {expected} at {_utc(ts[i]).isoformat()}"
                        )
                        if len(ret_problems) >= 6:
                            break
                    else:
                        ret_ok += 1

        if "norm_return_1" in df.columns and df["norm_return_1"].drop_nulls().len() >= 10:
            raw = [None]
            for i in range(1, len(close)):
                if close[i - 1] and close[i - 1] > 0 and close[i] is not None:
                    raw.append(close[i] / close[i - 1] - 1.0)
                else:
                    raw.append(None)
            lagged = [None] + raw[:-1]
            z = _rolling_z(lagged, 200)
            series = df["norm_return_1"].to_list()
            compared = 0
            for i, (got, expected) in enumerate(zip(series, z)):
                if expected is None or got is None:
                    continue
                if abs(got - expected) > 1e-6 * max(1.0, abs(expected)):
                    ret_problems.append(
                        f"{asset} norm_return_1 {got} != hand z {expected} at {_utc(ts[i]).isoformat()}"
                    )
                    if len(ret_problems) >= 6:
                        break
                else:
                    ret_ok += 1
                    compared += 1
                    if compared >= 200:
                        break

        if not _is_fx(asset):
            days = _trade_days(root, asset)
            if days and "wh_large_usd" in df.columns:
                daily_rows = []
                for day in days:
                    tpath = trades_path(root, asset, day)
                    trades = pl.read_parquet(tpath) if tpath.exists() else None
                    one_m = None
                    if trades is not None and not trades.is_empty() and asset in ladders and "time_1m" in ladders[asset]:
                        start = int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() * 1000)
                        end = start + DAY_MS
                        one_m = ladders[asset]["time_1m"].filter(
                            (pl.col("ts_ms") > start) & (pl.col("ts_ms") <= end)
                        )
                    daily_rows.append(
                        summarize_day(
                            trades, one_m, asset=asset, day=day,
                            whale_usd=whale_large_usd(asset), liq_usd=1e18,
                        )
                    )
                panel = rows_to_frame(daily_rows)
                if not panel.is_empty() and "wh_large_usd" in panel.columns:
                    events = list(zip(panel["event_time_ms"].to_list(), panel["wh_large_usd"].to_list()))
                    joined = df["wh_large_usd"].to_list()
                    for t, val in zip(ts, joined):
                        allowed = [v for ev, v in events if ev + DAY_MS <= t]
                        expected = allowed[-1] if allowed else None
                        if val is None and expected is None:
                            asof_ok += 1
                            continue
                        if val is None or expected is None or abs(float(val) - float(expected)) > 1e-4:
                            asof_problems.append(
                                f"{asset} wh_large_usd at {_utc(t).isoformat()} got={val} expected={expected}"
                            )
                            if len(asof_problems) >= 6:
                                break
                        else:
                            asof_ok += 1
                    for t, val in zip(ts, joined):
                        if val is None:
                            continue
                        future = [v for ev, v in events if ev + DAY_MS > t and v == val]
                        past = [v for ev, v in events if ev + DAY_MS <= t and v == val]
                        if future and not past:
                            asof_problems.append(
                                f"{asset} wh_large_usd={val} at {_utc(t).isoformat()} matches only a not-yet-knowable day"
                            )

        for col, source in (("stbl_supply_usd", "defillama"), ("etf_total_usdm", "farside")):
            if col not in df.columns:
                continue
            ext_dir = root / "bronze" / "external" / source
            files = sorted(ext_dir.glob("*.parquet")) if ext_dir.exists() else []
            if not files:
                continue
            ext = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
            if "asset" in ext.columns:
                own = ext.filter(pl.col("asset") == asset)
                if own.is_empty() and source == "defillama":
                    own = ext
            else:
                own = ext
            if own.is_empty() or "event_time_ms" not in own.columns or col not in own.columns:
                continue
            ev = list(zip(own["event_time_ms"].to_list(), own[col].to_list()))
            ev.sort()
            for t, val in zip(ts, df[col].to_list()):
                allowed = [v for e, v in ev if e + DAY_MS <= t]
                expected = allowed[-1] if allowed else None
                if val is None and expected is None:
                    asof_ok += 1
                    continue
                if expected is None and val is not None:
                    asof_problems.append(f"{asset} {col} present at {_utc(t).isoformat()} before any knowable row")
                    break
                if val is not None and expected is not None and abs(float(val) - float(expected)) > 1e-4 * max(1.0, abs(float(expected))):
                    asof_problems.append(
                        f"{asset} {col} at {_utc(t).isoformat()} got={val} expected={expected}"
                    )
                    break
                asof_ok += 1

    if ret_problems:
        report.add("hand_features", "FAIL", "; ".join(ret_problems[:5]))
    elif ret_ok == 0:
        report.add("hand_features", "WARN", "no finite returns to recompute")
    else:
        report.add("hand_features", "PASS", f"recomputed {ret_ok} target/trailing/z-scored returns by hand")
    if asof_problems:
        report.add("asof", "FAIL", "; ".join(asof_problems[:5]))
    else:
        report.add("asof", "PASS", f"{asof_ok} daily as-of values match event_time + 1 day lag")


def check_dead_columns(report: Report, gold_files: dict[str, dict[str, Path]]) -> None:
    allow = {
        "asset", "bar_type", "split", "regime_label", "is_u10", "is_u50", "is_u100",
        "is_warmup", "asset_dna", "listing_date", "bar_id",
        "is_tokyo", "is_london", "is_newyork", "is_ldn_ny_overlap",
    }
    funding_prefix = ("norm_funding", "funding_", "xd_funding")
    for bt, assets in sorted(gold_files.items()):
        for asset, path in sorted(assets.items()):
            df = _frame_for(path)
            dead = []
            const = []
            warmup = []
            for col in df.columns:
                if col in allow or col.startswith("target_"):
                    continue
                if col.startswith("etf_") and col != "etf_total_usdm":
                    # Issuer columns from the other ETF table stay null after a diagonal concat.
                    continue
                if col.startswith("dv_") and asset not in {"BTCUSDT", "ETHUSDT"}:
                    continue
                if col in {"xd_btc_return", "xd_btc_volatility"} and asset == "BTCUSDT":
                    # BTC's own return is not a cross-asset feature.
                    continue
                vals = df[col].drop_nulls()
                if vals.len() == 0:
                    if any(col.startswith(p) for p in funding_prefix):
                        warmup.append(f"{col}=all-null(funding unpublished)")
                    elif col == "norm_fd_close" and df.height < 1200:
                        warmup.append(f"{col}=warmup(frac_diff)")
                    elif (col.startswith("norm_") or col == "hurst_regime") and df.height < 400:
                        warmup.append(f"{col}=warmup")
                    elif col.startswith("xd_") and df.height < 30:
                        warmup.append(f"{col}=warmup")
                    elif _is_fx(asset) and col in FX_NULL_OK:
                        warmup.append(f"{col}=fx-null")
                    elif (lb := _lookback_bars(col)) is not None and df.height < lb:
                        warmup.append(f"{col}=warmup")
                    else:
                        dead.append(col)
                elif vals.len() >= 50 and vals.n_unique() == 1:
                    const.append(f"{col}={vals[0]}")
            name = f"columns:{asset}:{bt}"
            if dead:
                report.add(name, "WARN", f"all-null: {', '.join(dead[:8])}")
            elif const:
                report.add(name, "WARN", f"constant: {', '.join(const[:8])}")
            else:
                extra = f"; {len(warmup)} warmup/funding-null cols" if warmup else ""
                report.add(name, "PASS", f"{len(df.columns)} columns, no unexpected dead series{extra}")


def _vision_spot_1d_url(asset: str, day: date) -> str:
    return (
        f"https://data.binance.vision/data/spot/daily/klines/{asset}/1d/"
        f"{asset}-1d-{day.isoformat()}.zip"
    )


def check_external_closes(
    report: Report,
    root: Path,
    ladders: dict[str, dict[str, pl.DataFrame]],
    skip_network: bool,
) -> None:
    crypto_assets = [a for a in ladders if not _is_fx(a) and "time_1d" in ladders[a]]
    if crypto_assets and not skip_network:
        mismatches = []
        compared = 0
        samples = []
        for asset in crypto_assets[:4]:
            daily = ladders[asset]["time_1d"].sort("ts_ms")
            for row in daily.iter_rows(named=True):
                # Our 1d bar closing at D+1 00:00 is the UTC day D candle.
                close_ts = int(row["ts_ms"])
                day = _utc(close_ts - DAY_MS).date()
                url = _vision_spot_1d_url(asset, day)
                status, body, err = http_get(url, timeout=60)
                if status != 200 or not body:
                    mismatches.append(f"{asset} {day} vision HTTP {status or err}")
                    continue
                try:
                    k = parse_klines(body, width_ms=DAY_MS)
                except Exception as exc:  # noqa: BLE001
                    mismatches.append(f"{asset} {day} parse {exc}")
                    continue
                if k.is_empty():
                    mismatches.append(f"{asset} {day} empty kline")
                    continue
                venue_close = float(k["close"][0])
                our_close = float(row["close"])
                compared += 1
                if abs(venue_close - our_close) / max(our_close, 1e-9) > 1e-6:
                    mismatches.append(
                        f"{asset} {day} our_close={our_close} vision_1d={venue_close}"
                    )
                else:
                    samples.append(f"{asset} {day} close={our_close}")
        if compared == 0:
            report.add("crosscheck_crypto", "WARN", "could not fetch Vision spot 1d klines")
        elif mismatches:
            report.add("crosscheck_crypto", "FAIL", "; ".join(mismatches[:6]))
        else:
            report.add(
                "crosscheck_crypto", "PASS",
                f"{compared} daily closes match Vision spot 1d klines; e.g. " + "; ".join(samples[:3]),
            )
    elif crypto_assets and skip_network:
        # Offline: compare our 1d close to the last bronze 1m kline of that UTC day.
        mismatches = []
        compared = 0
        for asset in crypto_assets:
            daily = ladders[asset]["time_1d"].sort("ts_ms")
            for row in daily.iter_rows(named=True):
                day = _utc(int(row["ts_ms"]) - DAY_MS).date()
                path = klines_1m_path(root, asset, day)
                if not path.exists():
                    continue
                k = pl.read_parquet(path).sort("ts_ms")
                if k.is_empty():
                    continue
                venue_close = float(k["close"][-1])
                our_close = float(row["close"])
                compared += 1
                if abs(venue_close - our_close) / max(our_close, 1e-9) > 1e-5:
                    mismatches.append(f"{asset} {day} our={our_close} kline_1m_last={venue_close}")
        if mismatches:
            report.add("crosscheck_crypto", "FAIL", "; ".join(mismatches[:6]))
        elif compared:
            report.add("crosscheck_crypto", "PASS", f"{compared} daily closes match last Vision 1m kline of the day")
        else:
            report.add("crosscheck_crypto", "WARN", "no bronze 1m klines for an offline close check")

    fx_assets = [a for a in ladders if _is_fx(a) and "time_1d" in ladders[a]]
    if not fx_assets:
        return
    # Internal consistency is always reported; an external mid is best-effort.
    internal_ok = []
    for asset in fx_assets:
        days = _fx_tick_days(root, asset)
        ticks = _read_fx_ticks(root, asset, days)
        daily = ladders[asset]["time_1d"].sort("ts_ms")
        if ticks.is_empty():
            continue
        for row in daily.iter_rows(named=True):
            close_ts = int(row["ts_ms"])
            # Last tick in (open, close].
            window = ticks.filter((pl.col("ts_ms") > close_ts - DAY_MS) & (pl.col("ts_ms") <= close_ts))
            if window.is_empty():
                continue
            last_mid = float(window["mid"][-1])
            if abs(last_mid - float(row["close"])) > 1e-8:
                report.add(
                    f"crosscheck_fx:{asset}", "FAIL",
                    f"1d close {row['close']} != last mid in the UTC day {last_mid} "
                    f"ending {_utc(close_ts).date()}",
                )
                break
        else:
            internal_ok.append(asset)
    if internal_ok:
        report.add(
            "crosscheck_fx_internal", "PASS",
            f"1d mid close equals the last tick mid on {', '.join(internal_ok)}",
        )

    if skip_network or not fx_assets:
        return
    # Frankfurter (ECB reference, ~16:00 CET). Not a tick mid; report the pip gap, do not fail tightly.
    asset = "EURUSD" if "EURUSD" in fx_assets else fx_assets[0]
    daily = ladders[asset]["time_1d"].sort("ts_ms")
    if daily.is_empty():
        return
    # Compare the first complete weekday in the file.
    compared = []
    for row in daily.iter_rows(named=True):
        day = _utc(int(row["ts_ms"]) - DAY_MS).date()
        if day.weekday() >= 5:
            continue
        url = f"https://api.frankfurter.app/{day.isoformat()}?from=EUR&to=USD"
        if asset != "EURUSD":
            base, quote = asset[:3], asset[3:]
            url = f"https://api.frankfurter.app/{day.isoformat()}?from={base}&to={quote}"
        status, body, err = http_get(url, timeout=30)
        if status != 200 or not body:
            report.add("crosscheck_fx_external", "WARN", f"Frankfurter HTTP {status or err} for {day}")
            return
        try:
            payload = json.loads(body.decode("utf-8"))
            quote = asset[3:] if asset != "EURUSD" else "USD"
            ref = float(payload["rates"][quote])
        except Exception as exc:  # noqa: BLE001
            report.add("crosscheck_fx_external", "WARN", f"Frankfurter parse failed: {exc}")
            return
        mid = float(row["close"])
        pip = fx_pip(asset)
        gap = abs(mid - ref) / pip
        compared.append((day, mid, ref, gap))
        if len(compared) >= 3:
            break
    if compared:
        gaps = [c[3] for c in compared]
        # ECB ref vs Dukascopy end-of-UTC-day mid can be tens of pips. Report, warn only if huge.
        sev = "WARN" if max(gaps) > 100 else "PASS"
        detail = "; ".join(
            f"{d} dukascopy_1d={m:.5f} ECB={r:.5f} gap={g:.1f} pips" for d, m, r, g in compared
        )
        report.add("crosscheck_fx_external", sev, detail)


def load_ladders(found: dict) -> dict[str, dict[str, pl.DataFrame]]:
    ladders: dict[str, dict[str, pl.DataFrame]] = defaultdict(dict)
    layer = found["silver"] or found["gold"]
    for bt, assets in layer.items():
        for asset, path in assets.items():
            ladders[asset][bt] = _frame_for(path)
    return dict(ladders)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Independent validation of a pipeline data root")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--skip-network", action="store_true", help="skip Vision / Frankfurter cross-checks")
    args = p.parse_args(argv)
    root = args.data_root
    if not root.is_absolute():
        root = (_REPO / root).resolve()
    print(f"validate_run  data_root={root}")
    found = discover(root)
    gold_assets = sorted({a for bt in found["gold"].values() for a in bt})
    silver_assets = sorted({a for bt in found["silver"].values() for a in bt})
    print(
        f"  gold assets={len(gold_assets)} bar_types={sorted(found['gold'])}  "
        f"silver assets={len(silver_assets)} bar_types={sorted(found['silver'])}"
    )
    if not found["gold"] and not found["silver"]:
        print("  no gold or silver bars found")
        return 2

    ladders = load_ladders(found)
    # Frames used for monotonic checks: prefer gold (full dataset) when present.
    frames: dict[tuple[str, str], pl.DataFrame] = {}
    layer = found["gold"] or found["silver"]
    for bt, assets in layer.items():
        for asset, path in assets.items():
            frames[(asset, bt)] = _frame_for(path)

    report = Report()
    check_monotonic(report, frames)
    check_time_reconcile(report, ladders)
    check_trade_counts(report, root, ladders)
    check_gaps(report, ladders)
    check_sessions(report, found["gold"] or found["silver"])
    check_asof_and_hand_features(report, root, found["gold"] or found["silver"], ladders)
    if found["gold"]:
        check_dead_columns(report, found["gold"])
    check_external_closes(report, root, ladders, skip_network=args.skip_network)

    print()
    report.print()
    return report.code


if __name__ == "__main__":
    sys.exit(main())
