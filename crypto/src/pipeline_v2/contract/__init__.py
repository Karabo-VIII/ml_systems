"""Load the versioned contract files and the shared crypto config."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import yaml

from pipeline_v2.paths import crypto_root
from pipeline_v2.symbols import canonical_symbol

_CONTRACT = Path(__file__).resolve().parent


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
        return max(max(self.horizons), self.max_lookback)


def load_windows(path: Path | None = None) -> Windows:
    raw = _read(path or (_CONTRACT / "windows.yaml"))
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
    raw = _read(path or (_CONTRACT / "splits.yaml"))
    return Splits(
        train_end=_parse_date(raw["train_end"]),
        val_end=_parse_date(raw["val_end"]),
        oos_end=_parse_date(raw["oos_end"]),
        unseen_start=_parse_date(raw["unseen_start"]),
        live_start=_parse_date(raw["live_start"]),
    )


def load_dollar_thresholds(path: Path | None = None) -> dict:
    return _read(path or (_CONTRACT / "dollar_thresholds.yaml"))


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
    return _read(path or (_CONTRACT / "thresholds_daily.yaml"))


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


def load_universe(name: str) -> list[dict]:
    path = crypto_root() / "config" / "universes" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(path)
    data = _read(path)
    rows = list(data.get("assets") or [])
    if not rows and data.get("inherit_from"):
        rows = [
            {"symbol": a["symbol"], "dna": a.get("dna")}
            for a in load_universe(str(data["inherit_from"]))
        ]
    else:
        rows = [{"symbol": canonical_symbol(r["symbol"]), "dna": r.get("dna")} for r in rows]
    seen = {r["symbol"] for r in rows}
    for row in data.get("extra_assets") or []:
        sym = canonical_symbol(row["symbol"])
        if sym not in seen:
            rows.append({"symbol": sym, "dna": row.get("dna")})
            seen.add(sym)
    return rows


def universe_symbols(name: str) -> list[str]:
    return [a["symbol"] for a in load_universe(name)]


def load_listing_dates(path: Path | None = None) -> dict[str, date]:
    path = path or (crypto_root() / "config" / "asset_launch_dates.json")
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
