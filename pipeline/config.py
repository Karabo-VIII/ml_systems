"""Load the versioned contract files and the shared crypto config."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import yaml

from pipeline.paths import repo_root
from pipeline.symbols import canonical_fx_symbol, canonical_symbol


def config_file() -> Path:
    return repo_root() / "config" / "pipeline.yaml"


def _section(path: Path | None, key: str, marker: str) -> dict:
    """Read one section of config/pipeline.yaml, or a standalone section file."""
    raw = _read(path or config_file())
    if isinstance(raw, dict) and key in raw and marker not in raw:
        return raw[key]
    return raw


def _read(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


@dataclass(frozen=True)
class Windows:
    z: int = 200
    fast: int = 20
    vol_cluster: int = 50
    frac_diff: int = 1000
    frac_diff_d: float = 0.4
    hurst: int = 200
    kurtosis: int = 50
    efficiency: int = 10
    perm_entropy: int = 100
    perm_m: int = 3
    kyle: int = 50
    yz: int = 20
    cs_block: int = 10
    hawkes_span: int = 20
    ma: int = 200
    slow_span: int = 50
    # Every exponential average is evaluated over its last ``ema`` bars (weights
    # (1 - alpha)^k, normalised), so no feature has infinite memory. 240 bars drop
    # under 1e-8 of the weight of a span-26 EMA. Scaled down on coarse cadences.
    ema: int = 240
    # Rolling median that caps open interest before its change is taken (--include-oi).
    oi_median: int = 50
    horizons: tuple[int, ...] = (1, 4, 16, 64)

    def input_window(self) -> int:
        """Rows before a bar that the longest z-scored input reads, plus one.

        Every ``norm_*`` feature is ``rolling_z(x, z)`` of a series ``x`` built
        over one window, or over a chain of two: the EMA deviation followed by
        its rolling std (``vol_cluster``), one diff followed by the Hawkes EMA,
        and the open-interest median followed by the divergence EMAs. The total
        lookback is ``input_window + z``. 17 is ``_ret16`` (one shift plus 16).
        """
        plain = max(
            self.frac_diff, self.hurst, self.perm_entropy, self.kyle,
            self.vol_cluster, self.ma, self.kurtosis,
            self.yz, self.efficiency, self.fast,
            2 * self.cs_block, 17,
        ) + 1
        chained = max(
            self.ema + self.vol_cluster - 1,
            self.ema + 1,
            self.oi_median + self.ema,
        )
        return max(plain, chained)

    @property
    def max_lookback(self) -> int:
        """Longest trailing history any emitted column can touch (in bars).

        An input window followed by the ``z``-bar z-score, or the regime label's
        ``ma`` distance followed by its ``ma``-bar standard deviation.
        """
        return max(self.input_window() + self.z, 2 * self.ma + 1)

    def embargo_bars(self) -> int:
        """Flat bar count. Prefer embargo_bars_for(bar_type), which is cadence-aware."""
        return max(max(self.horizons), self.max_lookback)


def load_embargo_policy(path: Path | None = None) -> tuple[int, int]:
    raw = _section(path, "windows", "z")
    return int(raw.get("calendar_floor_days", 7)), int(raw.get("max_lookback_days", 90))


def bar_duration_ms(bar_type: str, median_bar_ms: int | None = None) -> int:
    """Clock width of one bar. Event bars use the observed median, else 5 minutes."""
    from pipeline.constants import TIME_WIDTH_MS
    if bar_type in TIME_WIDTH_MS:
        return TIME_WIDTH_MS[bar_type]
    if median_bar_ms and median_bar_ms > 0:
        return int(median_bar_ms)
    return 5 * 60_000


def scale_windows(
    windows: Windows,
    bar_type: str,
    median_bar_ms: int | None = None,
    *,
    max_lookback_days: int = 90,
) -> Windows:
    """Shrink windows so the *chained* lookback spans at most ``max_lookback_days``.

    A 1000-bar fractional difference is about 17 hours of 1-minute bars and
    1000 days of daily bars. Capping each window alone is not enough: every
    feature is an input window followed by a ``z``-bar z-score, so two 90-day
    windows chain to 180 days (the 2.3.0 defect). Here ``z`` takes at most half
    of the cap and every input window at most the rest, so
    ``max_lookback * width <= max_lookback_days``. Windows that already fit are
    left alone, which keeps fine cadences (1m to 1h) unchanged.
    """
    width = bar_duration_ms(bar_type, median_bar_ms)
    cap = max(4, int(max_lookback_days * 86_400_000 / width))
    if windows.max_lookback <= cap:
        return windows
    z = max(2, min(int(windows.z), cap // 2))
    # Input windows get what is left after z (and the +1 shift in input_window).
    rest = max(2, cap - z - 1)

    def cap_n(n: int, limit: int = rest, floor: int = 2) -> int:
        return max(floor, min(int(n), limit))

    # Chains of two windows share the budget: the deviation EMA feeds a rolling
    # std (vol_cluster), and the open-interest median feeds the divergence EMAs.
    ema = cap_n(windows.ema)
    vol_cluster = cap_n(windows.vol_cluster)
    if ema + vol_cluster - 1 > rest + 1:
        vol_cluster = max(2, min(vol_cluster, (rest + 2) // 2))
        ema = max(2, min(ema, rest + 2 - vol_cluster))
    oi_median = cap_n(windows.oi_median)
    if oi_median + ema > rest + 1:
        oi_median = max(2, rest + 1 - ema)
        if oi_median + ema > rest + 1:  # the EMA took the whole budget
            ema = max(2, rest + 1 - oi_median)

    scaled = Windows(
        z=z,
        fast=cap_n(windows.fast),
        vol_cluster=vol_cluster,
        frac_diff=cap_n(windows.frac_diff),
        frac_diff_d=windows.frac_diff_d,
        hurst=cap_n(windows.hurst),
        kurtosis=cap_n(windows.kurtosis),
        efficiency=cap_n(windows.efficiency),
        perm_entropy=cap_n(windows.perm_entropy),
        perm_m=windows.perm_m,
        kyle=cap_n(windows.kyle),
        yz=cap_n(windows.yz),
        cs_block=max(2, min(int(windows.cs_block), rest // 2)),
        hawkes_span=cap_n(windows.hawkes_span),
        # The regime label chains ma twice.
        ma=cap_n(windows.ma, min(rest, max(2, (cap - 1) // 2))),
        slow_span=cap_n(windows.slow_span),
        ema=ema,
        oi_median=oi_median,
        horizons=windows.horizons,
    )
    if scaled.max_lookback > cap:
        # Only on bars wider than about 2.5 days, where a fixed reach (the 16-bar
        # return) no longer fits beside half the cap: z gets what is left.
        import dataclasses
        scaled = dataclasses.replace(scaled, z=max(2, cap - scaled.input_window()))
    return scaled


def embargo_bars_for(
    bar_type: str,
    windows: Windows,
    median_bar_ms: int | None = None,
    *,
    calendar_floor_days: int = 7,
    max_lookback_days: int = 90,
) -> int:
    """Bars to drop at each split edge for this cadence.

    ``max(horizon, scaled feature lookback)`` in this cadence's bars, and at
    least ``calendar_floor_days`` of clock time. Not a flat 1200 for every file.
    """
    scaled = scale_windows(
        windows, bar_type, median_bar_ms, max_lookback_days=max_lookback_days,
    )
    width = bar_duration_ms(bar_type, median_bar_ms)
    floor_bars = max(1, math.ceil(calendar_floor_days * 86_400_000 / width))
    structural = max(max(scaled.horizons), scaled.max_lookback)
    return max(structural, floor_bars)


def load_windows(path: Path | None = None) -> Windows:
    raw = _section(path, "windows", "z")
    horizons = tuple(int(h) for h in raw.get("horizons", (1, 4, 16, 64)))
    fields = {k: raw[k] for k in (
        "z", "fast", "vol_cluster", "frac_diff", "frac_diff_d", "hurst",
        "kurtosis", "efficiency", "perm_entropy", "perm_m", "kyle", "yz",
        "cs_block", "hawkes_span", "ma", "slow_span", "ema", "oi_median",
    ) if k in raw}
    return Windows(horizons=horizons, **fields)


@dataclass(frozen=True)
class Splits:
    train_end: date
    val_end: date
    oos_end: date
    unseen_start: date
    live_start: date


def _parse_date(s: str) -> date:
    return datetime.strptime(s[:10], "%Y-%m-%d").date()


def load_splits(path: Path | None = None) -> Splits:
    raw = _section(path, "splits", "train_end")
    return Splits(
        train_end=_parse_date(raw["train_end"]),
        val_end=_parse_date(raw["val_end"]),
        oos_end=_parse_date(raw["oos_end"]),
        unseen_start=_parse_date(raw["unseen_start"]),
        live_start=_parse_date(raw["live_start"]),
    )


def load_lags(path: Path | None = None) -> dict:
    """The ``lags.families`` table: publication lag in ms per series family."""
    return dict(_section(path, "lags", "families")["families"])


def lag_ms(family: str, table: dict | None = None) -> int:
    """Publication lag of one series family. A family that is not listed is an error."""
    table = table if table is not None else load_lags()
    if family not in table:
        raise KeyError(f"no publication lag for {family!r} in config/pipeline.yaml lags.families")
    value = int(table[family])
    if value < 0:
        raise ValueError(f"lag for {family!r} must be >= 0, got {value}")
    return value


def load_dollar_thresholds(path: Path | None = None) -> dict:
    return _section(path, "dollar_thresholds", "calibration")


@dataclass(frozen=True)
class DollarSpec:
    """How dollar bars are sized for one asset.

    Adaptive (production): ``constant`` is ``None``. The size of UTC day D is the
    trailing median of daily notional before D over ``target_bars_per_day`` (see
    ``bars.trailing_threshold``). The first day of the chain has no history, so it
    is a calibration day: its trades seed the history and close no bar. No size
    is ever taken from a static table.

    Constant (tests and explicit overrides): ``target_bars_per_day`` is ``None``
    and every bar uses ``constant``.
    """

    constant: float | None
    target_bars_per_day: float | None
    window_days: int

    def as_params(self) -> dict:
        return {
            "constant": self.constant,
            "target_bars_per_day": self.target_bars_per_day,
            "window_days": self.window_days,
        }


def dollar_spec(asset: str, override: float | None = None, table: dict | None = None) -> DollarSpec:
    """Dollar-bar sizing. An explicit ``override`` means a constant threshold."""
    if override is not None:
        return DollarSpec(float(override), None, 30)
    table = table if table is not None else load_dollar_thresholds()
    cal = table.get("calibration") or {}
    if not isinstance(cal, dict) or not cal.get("target_bars_per_day"):
        raise KeyError("dollar_thresholds.calibration.target_bars_per_day is required in config/pipeline.yaml")
    return DollarSpec(None, float(cal["target_bars_per_day"]), int(cal.get("window_days", 30)))


def load_daily_thresholds(path: Path | None = None) -> dict:
    return _section(path, "daily_thresholds", "whale")


@dataclass(frozen=True)
class WhaleSpec:
    """Causal per-day whale size for ``wh_*`` features.

    The threshold of UTC day D is the ``quantile`` of per-trade notional over
    the last ``window_days`` observed complete days *before* D. Day D's own
    trades never move day D's size. With no history (an asset's first day) there
    is no size, and the day's ``wh_*`` columns are null.
    """

    quantile: float
    window_days: int
    version: str

    def as_params(self) -> dict:
        return {
            "quantile": self.quantile,
            "window_days": self.window_days,
            "version": self.version,
        }


def whale_spec(asset: str, table: dict | None = None) -> WhaleSpec:
    table = table if table is not None else load_daily_thresholds()
    cal = table.get("whale") or {}
    return WhaleSpec(
        float(cal.get("quantile", 0.999)),
        int(cal.get("window_days", 30)),
        str(table.get("version", "daily_thresholds_v3")),
    )


def load_fx() -> dict:
    return _read(config_file())["fx"]


def fx_price_divisor(pair: str) -> float:
    table = load_fx()["price_divisor"]
    return float(table["jpy"] if "JPY" in pair.upper() else table["default"])


def fx_pip(pair: str) -> float:
    table = load_fx()["pip"]
    return float(table["jpy"] if "JPY" in pair.upper() else table["default"])


def load_universe(name: str) -> list[dict]:
    universes = _read(config_file()).get("universes") or {}
    if name not in universes:
        raise FileNotFoundError(f"universe {name} is not in config/pipeline.yaml")
    data = universes[name]
    market = data.get("market")

    def canon(sym: str) -> str:
        if market == "fx":
            return canonical_fx_symbol(sym)
        return canonical_symbol(sym)

    rows = list(data.get("assets") or [])
    if not rows and data.get("inherit_from"):
        rows = [
            {"symbol": a["symbol"], "dna": a.get("dna")}
            for a in load_universe(str(data["inherit_from"]))
        ]
    else:
        rows = [{"symbol": canon(r["symbol"]), "dna": r.get("dna")} for r in rows]
    seen = {r["symbol"] for r in rows}
    for row in data.get("extra_assets") or []:
        sym = canon(row["symbol"])
        if sym not in seen:
            rows.append({"symbol": sym, "dna": row.get("dna")})
            seen.add(sym)
    return rows


def universe_symbols(name: str) -> list[str]:
    return [a["symbol"] for a in load_universe(name)]


def load_listing_dates(path: Path | None = None) -> dict[str, date]:
    path = path or (repo_root() / "config" / "asset_launch_dates.json")
    if not path.exists():
        return {}
    import json
    raw = json.loads(path.read_text())
    out: dict[str, date] = {}
    for k, v in raw.items():
        try:
            out[canonical_symbol(k)] = _parse_date(str(v))
        except ValueError:
            continue
    return out


def effective_start(asset: str, start: date, listings: dict[str, date] | None = None) -> date:
    """History starts at max(requested start, 2020-01-01, listing date)."""
    floor = _parse_date("2020-01-01")
    listings = listings if listings is not None else load_listing_dates()
    listed = listings.get(canonical_symbol(asset), floor)
    return max(start, floor, listed)


def membership_flags(asset: str) -> dict[str, bool | str | None]:
    asset = canonical_symbol(asset)
    flags: dict[str, bool | str | None] = {
        "is_u10": False, "is_u50": False, "is_u100": False, "asset_dna": None,
    }
    for name, key in (("u10", "is_u10"), ("u50", "is_u50"), ("u100", "is_u100")):
        try:
            rows = load_universe(name)
        except (FileNotFoundError, KeyError, TypeError):
            continue
        for row in rows:
            if row["symbol"] == asset:
                flags[key] = True
                if flags["asset_dna"] is None and row.get("dna"):
                    flags["asset_dna"] = row["dna"]
    return flags
