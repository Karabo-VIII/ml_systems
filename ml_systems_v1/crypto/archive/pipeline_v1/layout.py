"""Canonical data layout resolver (post-2026-04-26 cleanup, v3 = cadence subfolders).

Centralizes path construction + date-suffix logic so producers + readers stay
in sync. Per-instrument files are qualified by their data-end date so freshness
is visible at a glance, AND grouped under cadence/type subfolders for homogeneous
storage.

    data/processed/chimera/<cadence>/<sym>usdt_v51_chimera[_<cad>]_<YYYYMMDD>.parquet
        cadences: dollar/, 1d/, 4h/, 1h/, 30m/, 15m/
    data/processed/chimera_legacy/dollar/<sym>usdt_v50_chimera_<YYYYMMDD>.parquet
    data/processed/frontier/daily/<sym>usdt_frontier_daily_<YYYYMMDD>.parquet
    data/processed/bars/<bartype>/<sym>usdt_<bartype>_<YYYYMMDD>.parquet
        bartypes: dib/, runs_tick/, runs_volume/, range/, adaptive_vol/
    data/processed/hawkes/daily/<panel>_<YYYYMMDD>.parquet
    data/processed/panels/daily/<panel>_<YYYYMMDD>.parquet

Multi-asset panels (no per-instrument key) use just the panel name + date.

Reader semantics: glob the cadence/type directory for `<prefix>_<YYYYMMDD>.parquet`
and pick the lexicographically largest filename — since the date suffix is
YYYYMMDD, lexicographic = chronological. Builders always write a new dated file
and gc older same-key files in the same subfolder.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT_ROOT / "data"
PROCESSED = DATA / "processed"

# Top-level layer dirs under processed/
DIR_CHIMERA = PROCESSED / "chimera"            # v51 SOTA gold (cadence subfolders)
DIR_CHIMERA_LEGACY = PROCESSED / "chimera_legacy"   # v50 legacy (dollar/ subfolder)
DIR_FRONTIER = PROCESSED / "frontier"          # silver per-asset (daily/ subfolder)
DIR_BARS = PROCESSED / "bars"                   # silver bar fabric (bartype subfolders)
DIR_HAWKES = PROCESSED / "hawkes"              # multi-asset Hawkes (daily/ subfolder)
DIR_PANELS = PROCESSED / "panels"              # multi-asset wide silver (daily/ subfolder)
DIR_MANIFESTS = DATA / "manifests"             # per-build lineage

# Cadence keys for v51 chimera (also serve as subfolder names)
# Time cadences (dollar + time-resamples) AND alt chart types (2026-05-29 L2:
# feature-enriched alt-bar chimeras live under the SAME chimera/<x>/ layout, so
# they are addressable as first-class "cadences" by the loader).
V51_CADENCES = ("dollar", "1d", "4h", "1h", "30m", "15m",
                "dib", "runs_tick", "runs_volume", "range", "adaptive_vol")
V51_CADENCE_TAG = {
    "dollar": "",       # no cadence suffix in filename for primary dollar bars
    "1d": "_1d",
    "4h": "_4h",
    "1h": "_1h",
    "30m": "_30m",
    "15m": "_15m",
    # alt chart types: filename tag = _<bartype> (matches make_chimera_bars.py output)
    "dib": "_dib",
    "runs_tick": "_runs_tick",
    "runs_volume": "_runs_volume",
    "range": "_range",
    "adaptive_vol": "_adaptive_vol",
}

# Bar fabric types (also serve as subfolder names)
BAR_TYPES = ("dib", "runs_tick", "runs_volume", "range", "adaptive_vol")


def date_yyyymmdd(d: date | datetime) -> str:
    if isinstance(d, datetime):
        d = d.date()
    return d.strftime("%Y%m%d")


def latest_date_from_ts_ms(ts_ms_max: int) -> date:
    return datetime.fromtimestamp(ts_ms_max / 1000.0, tz=timezone.utc).date()


def normalize_asset(asset: str) -> tuple[str, str]:
    """Return (asset_l_with_usdt, asset_u_with_usdt) — e.g. ('btcusdt', 'BTCUSDT')."""
    a = asset.upper()
    if a.endswith("USDT"):
        root = a[:-4]
    else:
        root = a
    return f"{root.lower()}usdt", f"{root}USDT"


# ─── Subfolder accessors (per-layer, per-cadence/type) ──────────────────────

def chimera_dir(cadence: str) -> Path:
    """data/processed/chimera/<cadence>/"""
    if cadence not in V51_CADENCES:
        raise ValueError(f"unknown cadence {cadence!r}; known={V51_CADENCES}")
    return DIR_CHIMERA / cadence


def chimera_legacy_dir() -> Path:
    """data/processed/chimera_legacy/dollar/  (legacy v50 = dollar bars only)"""
    return DIR_CHIMERA_LEGACY / "dollar"


def frontier_dir() -> Path:
    """data/processed/frontier/daily/  (silver per-asset = daily only)"""
    return DIR_FRONTIER / "daily"


def bars_dir(bartype: str) -> Path:
    """data/processed/bars/<bartype>/

    Tolerant of legacy short-name dir for runs_volume (Tier 1C deferred):
    actual on-disk dir is sometimes `runs_vol/` instead of canonical
    `runs_volume/`. Returns whichever exists; canonical wins if both do.
    """
    if bartype not in BAR_TYPES:
        raise ValueError(f"unknown bartype {bartype!r}; known={BAR_TYPES}")
    canonical = DIR_BARS / bartype
    if canonical.exists():
        return canonical
    if bartype == "runs_volume":
        legacy = DIR_BARS / "runs_vol"
        if legacy.exists():
            return legacy
    return canonical


def hawkes_dir() -> Path:
    """data/processed/hawkes/daily/  (multi-asset Hawkes panels = daily only)"""
    return DIR_HAWKES / "daily"


def panels_dir() -> Path:
    """data/processed/panels/daily/  (multi-asset wide panels = daily only)"""
    return DIR_PANELS / "daily"


# ─── Write paths (producers call these) ──────────────────────────────────────

def chimera_v51_path(asset: str, cadence: str, latest_date: date | None = None) -> Path:
    """data/processed/chimera/<cadence>/<sym>usdt_v51_chimera[_<cad>]_<YYYYMMDD>.parquet"""
    if cadence not in V51_CADENCES:
        raise ValueError(f"unknown cadence {cadence!r}; known={V51_CADENCES}")
    sym_l, _ = normalize_asset(asset)
    cad_tag = V51_CADENCE_TAG[cadence]
    d = chimera_dir(cadence)
    if latest_date is None:
        return d / f"{sym_l}_v51_chimera{cad_tag}.parquet"
    return d / f"{sym_l}_v51_chimera{cad_tag}_{date_yyyymmdd(latest_date)}.parquet"


def chimera_v50_path(asset: str, latest_date: date | None = None) -> Path:
    """data/processed/chimera_legacy/dollar/<sym>usdt_v50_chimera_<YYYYMMDD>.parquet"""
    sym_l, _ = normalize_asset(asset)
    d = chimera_legacy_dir()
    if latest_date is None:
        return d / f"{sym_l}_v50_chimera.parquet"
    return d / f"{sym_l}_v50_chimera_{date_yyyymmdd(latest_date)}.parquet"


def frontier_daily_path(asset: str, latest_date: date | None = None) -> Path:
    """data/processed/frontier/daily/<sym>usdt_frontier_daily_<YYYYMMDD>.parquet"""
    sym_l, _ = normalize_asset(asset)
    d = frontier_dir()
    if latest_date is None:
        return d / f"{sym_l}_frontier_daily.parquet"
    return d / f"{sym_l}_frontier_daily_{date_yyyymmdd(latest_date)}.parquet"


def bars_path(asset: str, bartype: str, latest_date: date | None = None) -> Path:
    """data/processed/bars/<bartype>/<sym>usdt_<bartype>_<YYYYMMDD>.parquet"""
    sym_l, _ = normalize_asset(asset)
    d = bars_dir(bartype)
    if latest_date is None:
        return d / f"{sym_l}_{bartype}.parquet"
    return d / f"{sym_l}_{bartype}_{date_yyyymmdd(latest_date)}.parquet"


def hawkes_panel_path(name: str, latest_date: date | None = None) -> Path:
    """data/processed/hawkes/daily/<name>_<YYYYMMDD>.parquet"""
    d = hawkes_dir()
    if latest_date is None:
        return d / f"{name}.parquet"
    return d / f"{name}_{date_yyyymmdd(latest_date)}.parquet"


def panel_path(name: str, latest_date: date | None = None) -> Path:
    """data/processed/panels/daily/<name>_<YYYYMMDD>.parquet"""
    d = panels_dir()
    if latest_date is None:
        return d / f"{name}.parquet"
    return d / f"{name}_{date_yyyymmdd(latest_date)}.parquet"


def manifest_path(asset: str) -> Path:
    """data/manifests/v51_<SYM>.json (manifests don't carry date in name; build_date inside)."""
    _, sym_u = normalize_asset(asset)
    return DIR_MANIFESTS / f"v51_{sym_u}.json"


# ─── Read paths (consumers call these) — pick latest dated file ──────────────

def _pick_latest(directory: Path, prefix: str, suffix: str = ".parquet") -> Path | None:
    """Glob exactly `<directory>/<prefix>_<YYYYMMDD><suffix>` and return the largest.

    Strictly requires the date suffix to FOLLOW the prefix directly (no extra tags
    between prefix and date). Avoids cross-matching e.g. cadence files when looking
    up dollar bars.

    Since dates are YYYYMMDD-formatted, lex = chrono order. If no dated files
    exist, falls back to undated `<prefix><suffix>` if present.
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


def chimera_v51_latest(asset: str, cadence: str) -> Path | None:
    """Latest dated v51 chimera at this cadence, or None if none."""
    if cadence not in V51_CADENCES:
        raise ValueError(f"unknown cadence {cadence!r}")
    sym_l, _ = normalize_asset(asset)
    cad_tag = V51_CADENCE_TAG[cadence]
    return _pick_latest(chimera_dir(cadence), f"{sym_l}_v51_chimera{cad_tag}")


def chimera_v50_latest(asset: str) -> Path | None:
    sym_l, _ = normalize_asset(asset)
    return _pick_latest(chimera_legacy_dir(), f"{sym_l}_v50_chimera")


def frontier_daily_latest(asset: str) -> Path | None:
    sym_l, _ = normalize_asset(asset)
    return _pick_latest(frontier_dir(), f"{sym_l}_frontier_daily")


def bars_latest(asset: str, bartype: str) -> Path | None:
    """Latest dated bar file at <bars_dir>/<sym>_<bartype>_<YYYYMMDD>.parquet.

    Tolerant of the legacy bartype-writer outputs (Tier 1C deferred migration):
      - UPPERCASE asset prefix (e.g. AAVEUSDT_dib_2025.parquet)
      - 4-char "_2025" suffix instead of 8-digit YYYYMMDD
      - undated patterns (e.g. AAVEUSDT_adaptive_vol.parquet)
      - swapped runs_<mode> vs <mode>_runs naming
    Returns None only if no file (canonical or legacy) exists for the asset.
    """
    sym_l, sym_u = normalize_asset(asset)
    d = bars_dir(bartype)
    canonical = _pick_latest(d, f"{sym_l}_{bartype}")
    if canonical is not None:
        return canonical

    if not d.exists():
        return None

    # Legacy fallback: case-insensitive prefix scan, also handle
    # runs_tick <-> tick_runs / runs_vol <-> vol_runs writer-vs-reader name swap.
    legacy_aliases = {bartype}
    if bartype == "runs_tick":
        legacy_aliases.add("tick_runs")
    elif bartype == "runs_vol":
        legacy_aliases.add("vol_runs")
    elif bartype == "runs_volume":
        legacy_aliases.update({"vol_runs", "runs_vol"})

    candidates: list[Path] = []
    for f in d.glob("*.parquet"):
        stem_lower = f.stem.lower()
        for alias in legacy_aliases:
            # Match either "<sym><alias>" or "<sym>_<alias>" prefix
            if stem_lower.startswith(f"{sym_l}_{alias}"):
                candidates.append(f)
                break
    if not candidates:
        return None
    # Lex-sort: canonical YYYYMMDD-suffixed names beat _2025 / undated, and
    # within same prefix the largest stem wins (latest date).
    return sorted(candidates, key=lambda p: p.stem)[-1]


def hawkes_panel_latest(name: str) -> Path | None:
    return _pick_latest(hawkes_dir(), name)


def panel_latest(name: str) -> Path | None:
    return _pick_latest(panels_dir(), name)


# ─── GC: keep only the newest dated file per (dir, prefix) ───────────────────

def gc_older_dated(directory: Path, prefix: str, keep_newest: int = 1,
                   suffix: str = ".parquet", dry_run: bool = False) -> list[Path]:
    """Delete all but the newest `keep_newest` dated files matching prefix.

    Strictly requires `<prefix>_<YYYYMMDD><suffix>` (no extra tags between prefix
    and date). Returns the list of files that were (or would be, if dry_run) deleted.
    """
    if not directory.exists():
        return []
    expected_prefix = f"{prefix}_"
    dated = []
    for f in directory.glob(f"{prefix}_*{suffix}"):
        stem = f.stem
        if not stem.startswith(expected_prefix):
            continue
        tail = stem[len(expected_prefix):]
        if tail.isdigit() and len(tail) == 8:
            dated.append(f)
    dated = sorted(dated)
    if len(dated) <= keep_newest:
        return []
    to_delete = dated[:-keep_newest]
    deleted = []
    if not dry_run:
        for f in to_delete:
            try:
                f.unlink()
                deleted.append(f)
            except OSError as e:
                # Do not silently report a still-present file as deleted (e.g.
                # Windows file-in-use). Warn and exclude it from the return list.
                print(f"[layout] WARN gc could not delete {f.name}: "
                      f"{type(e).__name__}: {e}", flush=True)
        return deleted
    return to_delete


def list_v51_assets() -> list[str]:
    """Discover all assets that have a v51 dollar-bar chimera in the new layout."""
    d = chimera_dir("dollar")
    if not d.exists():
        return []
    out = set()
    for f in d.glob("*_v51_chimera_*.parquet"):
        stem = f.stem
        if "_" not in stem:
            continue
        without_date = stem.rsplit("_", 1)
        date_part = without_date[1]
        if not (date_part.isdigit() and len(date_part) == 8):
            continue
        prefix_part = without_date[0]
        if prefix_part.endswith("_v51_chimera"):
            sym_l = prefix_part[: -len("_v51_chimera")]
            out.add(sym_l.upper())
    return sorted(out)


def list_v50_assets() -> list[str]:
    """Discover all assets that have a v50 chimera in chimera_legacy/dollar/."""
    d = chimera_legacy_dir()
    if not d.exists():
        return []
    out = set()
    for f in d.glob("*_v50_chimera_*.parquet"):
        stem = f.stem
        if "_" not in stem:
            continue
        without_date = stem.rsplit("_", 1)
        date_part = without_date[1]
        if not (date_part.isdigit() and len(date_part) == 8):
            continue
        prefix_part = without_date[0]
        if prefix_part.endswith("_v50_chimera"):
            sym_l = prefix_part[: -len("_v50_chimera")]
            out.add(sym_l.upper())
    return sorted(out)


@dataclass(frozen=True)
class LayoutVersion:
    """Marker so callers can be sure they're using the post-2026-04-26 layout."""
    name: str = "v3"
    canonical_date: str = "2026-04-26"


VERSION = LayoutVersion()
