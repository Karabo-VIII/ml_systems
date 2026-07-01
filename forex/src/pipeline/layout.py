"""Canonical forex data-layout resolver + pair-name normalization.

Centralizes path construction and pair-name conversions so producers and
readers stay in sync. Analogous to the crypto pipeline's layout.py, adapted
for FX (no dollar bars; time-cadence bars from ticks; bid/ask first-class).

    data/raw/<PAIR>/ticks/<pair>_ticks_<YYYY-MM-DD>.parquet     # Dukascopy (bronze)
    data/raw/<PAIR>/m1/<pair>_m1_<YYYY-MM>.parquet              # HistData M1 (bronze)
    data/processed/bars/<cadence>/<pair>_<cadence>_<YYYYMMDD>.parquet
        cadences: 1m/, 5m/, 15m/, 30m/, 1h/, 4h/, 1d/
    data/manifests/<PAIR>_fetch.json                            # per-pair lineage

Reader semantics: glob the cadence directory for `<prefix>_<YYYYMMDD>.parquet`
and pick the lexicographically largest filename (YYYYMMDD => lex = chrono).

PAIR is stored flat + UPPER on disk (EURUSD); the slash form (EUR/USD),
Dukascopy form (EURUSD), and OANDA form (EUR_USD) are derived via
`normalize_pair`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent   # forex/
DATA = PROJECT_ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
MANIFESTS = DATA / "manifests"

__contract__ = {
    "kind": "framework_helper",
    "stage": "layout",
    "outputs": {"paths": "canonical data/raw + data/processed resolver; pair-name forms"},
    "invariants": {
        "pair_normalization_6char": True,
        "date_suffix_yyyymmdd_lex_is_chrono": True,
        "flat_upper_on_disk_key": True,
    },
    "rationale": "Single source of truth for on-disk paths + pair-name conversions.",
}

DIR_BARS = PROCESSED / "bars"          # silver: time bars (per pair, per cadence)
DIR_FEATURES = PROCESSED / "features"  # silver: per-pair feature frames
DIR_PANELS = PROCESSED / "panels"      # silver: cross-pair panels (usd strength, corr, ...)
DIR_DATASET = PROCESSED / "dataset"    # gold: model-ready (bars + features + panel + targets)

# Time-bar cadences (also serve as subfolder names). No dollar bars in FX
# (spot FX has no consolidated volume) -> time bars are primary.
CADENCES = ("1m", "5m", "15m", "30m", "1h", "4h", "1d")

# Alternative (information-driven) bar types, built later off ticks.
BAR_TYPES = ("tick_count", "range")


def date_yyyymmdd(d: date | datetime) -> str:
    if isinstance(d, datetime):
        d = d.date()
    return d.strftime("%Y%m%d")


# ─── Pair-name normalization ────────────────────────────────────────────────

@dataclass(frozen=True)
class PairNames:
    slash: str      # EUR/USD
    flat: str       # EURUSD   (canonical on-disk key, UPPER)
    lower: str      # eurusd   (filename prefix)
    dukas: str      # EURUSD   (Dukascopy instrument)
    oanda: str      # EUR_USD  (OANDA v20 instrument)
    base: str       # EUR
    quote: str      # USD


def normalize_pair(pair: str) -> PairNames:
    """Accept EUR/USD | EURUSD | eur_usd | eurusd and return every form.

    Assumes 3-letter base + 3-letter quote (true for all FX majors/crosses/
    exotics and XAU/XAG metals). Raises ValueError on an unparseable input.
    """
    p = pair.upper().replace("/", "").replace("_", "").strip()
    if len(p) != 6:
        raise ValueError(
            f"normalize_pair: expected 6-char pair (e.g. EURUSD); got {pair!r} -> {p!r}")
    base, quote = p[:3], p[3:]
    return PairNames(
        slash=f"{base}/{quote}", flat=p, lower=p.lower(),
        dukas=p, oanda=f"{base}_{quote}", base=base, quote=quote,
    )


# ─── Raw (bronze) paths ─────────────────────────────────────────────────────

def ticks_dir(pair: str) -> Path:
    """data/raw/<PAIR>/ticks/"""
    n = normalize_pair(pair)
    return RAW / n.flat / "ticks"


def tick_day_path(pair: str, d: date | datetime) -> Path:
    """data/raw/<PAIR>/ticks/<pair>_ticks_<YYYY-MM-DD>.parquet (one file / UTC day)."""
    n = normalize_pair(pair)
    if isinstance(d, datetime):
        d = d.date()
    return ticks_dir(pair) / f"{n.lower}_ticks_{d.isoformat()}.parquet"


def m1_dir(pair: str) -> Path:
    """data/raw/<PAIR>/m1/  (HistData M1 OHLC, one file / month)."""
    n = normalize_pair(pair)
    return RAW / n.flat / "m1"


def m1_month_path(pair: str, year: int, month: int) -> Path:
    """data/raw/<PAIR>/m1/<pair>_m1_<YYYY-MM>.parquet"""
    n = normalize_pair(pair)
    return m1_dir(pair) / f"{n.lower}_m1_{year:04d}-{month:02d}.parquet"


# ─── Processed (silver) bar paths ───────────────────────────────────────────

def bars_dir(cadence: str) -> Path:
    """data/processed/bars/<cadence>/"""
    if cadence not in CADENCES and cadence not in BAR_TYPES:
        raise ValueError(f"unknown cadence {cadence!r}; known={CADENCES + BAR_TYPES}")
    return DIR_BARS / cadence


def bars_path(pair: str, cadence: str, latest_date: date | None = None) -> Path:
    """data/processed/bars/<cadence>/<pair>_<cadence>_<YYYYMMDD>.parquet"""
    n = normalize_pair(pair)
    d = bars_dir(cadence)
    if latest_date is None:
        return d / f"{n.lower}_{cadence}.parquet"
    return d / f"{n.lower}_{cadence}_{date_yyyymmdd(latest_date)}.parquet"


# ─── Reader helpers (pick latest dated file) ────────────────────────────────

def _pick_latest(directory: Path, prefix: str, suffix: str = ".parquet") -> Path | None:
    """Return the lexicographically-largest `<prefix>_<YYYYMMDD><suffix>` file.

    Requires the 8-digit date to directly follow the prefix. Falls back to an
    undated `<prefix><suffix>` if no dated file exists.
    """
    if not directory.exists():
        return None
    dated = []
    expected_prefix = f"{prefix}_"
    for f in directory.glob(f"{prefix}_*{suffix}"):
        stem = f.stem
        if not stem.startswith(expected_prefix):
            continue
        tail = stem[len(expected_prefix):]
        if tail.isdigit() and len(tail) == 8:
            dated.append(f)
    if dated:
        return sorted(dated)[-1]
    fallback = directory / f"{prefix}{suffix}"
    return fallback if fallback.exists() else None


def bars_latest(pair: str, cadence: str) -> Path | None:
    n = normalize_pair(pair)
    return _pick_latest(bars_dir(cadence), f"{n.lower}_{cadence}")


def list_tick_days(pair: str) -> list[date]:
    """All UTC dates for which a raw tick file exists for this pair."""
    d = ticks_dir(pair)
    if not d.exists():
        return []
    n = normalize_pair(pair)
    out: list[date] = []
    prefix = f"{n.lower}_ticks_"
    for f in d.glob(f"{prefix}*.parquet"):
        tail = f.stem[len(prefix):]
        try:
            out.append(date.fromisoformat(tail))
        except ValueError:
            continue
    return sorted(out)


def features_dir(cadence: str) -> Path:
    return DIR_FEATURES / cadence


def features_path(pair: str, cadence: str, latest_date: date | None = None) -> Path:
    """data/processed/features/<cadence>/<pair>_features_<YYYYMMDD>.parquet"""
    n = normalize_pair(pair)
    d = features_dir(cadence)
    if latest_date is None:
        return d / f"{n.lower}_features.parquet"
    return d / f"{n.lower}_features_{date_yyyymmdd(latest_date)}.parquet"


def features_latest(pair: str, cadence: str) -> Path | None:
    n = normalize_pair(pair)
    return _pick_latest(features_dir(cadence), f"{n.lower}_features")


def panel_path(name: str, cadence: str, latest_date: date | None = None) -> Path:
    """data/processed/panels/<cadence>/<name>_<YYYYMMDD>.parquet (multi-pair; no pair key)."""
    d = DIR_PANELS / cadence
    if latest_date is None:
        return d / f"{name}.parquet"
    return d / f"{name}_{date_yyyymmdd(latest_date)}.parquet"


def panel_latest(name: str, cadence: str) -> Path | None:
    return _pick_latest(DIR_PANELS / cadence, name)


def dataset_path(pair: str, cadence: str, latest_date: date | None = None) -> Path:
    """data/processed/dataset/<cadence>/<pair>_dataset_<YYYYMMDD>.parquet (gold, model-ready)."""
    n = normalize_pair(pair)
    d = DIR_DATASET / cadence
    if latest_date is None:
        return d / f"{n.lower}_dataset.parquet"
    return d / f"{n.lower}_dataset_{date_yyyymmdd(latest_date)}.parquet"


def dataset_latest(pair: str, cadence: str) -> Path | None:
    n = normalize_pair(pair)
    return _pick_latest(DIR_DATASET / cadence, f"{n.lower}_dataset")


def manifest_path(pair: str) -> Path:
    """data/manifests/<PAIR>_fetch.json"""
    n = normalize_pair(pair)
    return MANIFESTS / f"{n.flat}_fetch.json"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class LayoutVersion:
    name: str = "fx_v1"
    canonical_date: str = "2026-07-01"


VERSION = LayoutVersion()
