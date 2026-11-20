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
    funding_mom: int = 30
    hawkes_span: int = 20
    ma: int = 200
    horizons: tuple[int, ...] = (1, 4, 16, 64)

    @property
    def max_lookback(self) -> int:
        """Longest trailing history any enabled default feature can touch.

        Fractional difference at t uses ``frac_diff`` bars, then the z-score
        uses another ``z`` bars of that series.
        """
        return max(
            self.frac_diff + self.z,
            self.hurst + self.z,
            self.perm_entropy + self.z,
            self.kyle + self.z,
            self.vol_cluster + self.z,
            self.ma + self.z,
            self.z,
        )

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
    """Shrink trailing windows so none spans more than ``max_lookback_days``.

    A 1000-bar fractional difference is about 17 hours of 1-minute bars and
    1000 days of daily bars. The cap keeps the daily lookback on a calendar
    scale. Windows that are already shorter are left alone.
    """
    width = bar_duration_ms(bar_type, median_bar_ms)
    cap = max(int(windows.fast), int(max_lookback_days * 86_400_000 / width))

    def cap_n(n: int, floor: int = 2) -> int:
        return max(floor, min(int(n), cap))

    return Windows(
        z=cap_n(windows.z),
        fast=min(int(windows.fast), cap_n(windows.fast)),
        vol_cluster=cap_n(windows.vol_cluster),
        frac_diff=cap_n(windows.frac_diff),
        frac_diff_d=windows.frac_diff_d,
        hurst=cap_n(windows.hurst),
        kurtosis=cap_n(windows.kurtosis),
        efficiency=min(int(windows.efficiency), cap_n(windows.efficiency)),
        perm_entropy=cap_n(windows.perm_entropy),
        perm_m=windows.perm_m,
        kyle=cap_n(windows.kyle),
        yz=cap_n(windows.yz),
        cs_block=max(2, min(int(windows.cs_block), max(2, cap // 4))),
        funding_mom=cap_n(windows.funding_mom),
        hawkes_span=min(int(windows.hawkes_span), cap_n(windows.hawkes_span)),
        ma=cap_n(windows.ma),
        horizons=windows.horizons,
    )


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
        "cs_block", "funding_mom", "hawkes_span", "ma",
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


def load_dollar_thresholds(path: Path | None = None) -> dict:
    return _section(path, "dollar_thresholds", "assets")


def dollar_threshold(asset: str, table: dict | None = None) -> float:
    table = table if table is not None else load_dollar_thresholds()
    asset = canonical_symbol(asset)
    assets = table["assets"]
    if asset not in assets:
        raise KeyError(
            f"no dollar threshold for {asset} in dollar_thresholds.yaml "
            f"(version {table.get('version')}). Add one; do not invent a size."
        )
    return float(assets[asset])


def optional_threshold(asset: str, kind: str, table: dict | None = None) -> float:
    table = table if table is not None else load_dollar_thresholds()
    asset = canonical_symbol(asset)
    block = table[kind]
    if asset in block:
        return float(block[asset])
    return float(block["default"])


def load_daily_thresholds(path: Path | None = None) -> dict:
    return _section(path, "daily_thresholds", "whale_usd")


def whale_large_usd(asset: str, table: dict | None = None) -> float:
    table = table if table is not None else load_daily_thresholds()
    asset = canonical_symbol(asset)
    pair = table["whale_usd"].get(asset, table["whale_usd"]["default"])
    return float(pair[1])


def liq_proxy_usd(asset: str, table: dict | None = None) -> float:
    table = table if table is not None else load_daily_thresholds()
    asset = canonical_symbol(asset)
    block = table["liq_proxy_usd"]
    return float(block.get(asset, block["default"]))


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
