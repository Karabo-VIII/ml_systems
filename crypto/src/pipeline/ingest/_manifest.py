"""Shared confirmed-missing manifest helper for pipeline ingesters.

Extracted from binance_vision_book_depth_profile.py (commit fb128e5, 2026-05-24)
so that every web ingester can use the same skip-on-404 pattern without
duplicating the atomic-write + per-asset-lock machinery.

Usage pattern
-------------
    from pipeline.ingest._manifest import MissingManifest

    mm = MissingManifest(out_root=OUT_DIR / sym)          # per-asset root
    if mm.is_known_missing(sym, date_str):
        return SKIP
    raw = fetch(url)
    if raw is None:
        mm.mark_missing(sym, date_str)
        return NOT_AVAILABLE
    mm.unmark_missing(sym, date_str)                      # in case it was stale
    ...

Manifest schema (per asset, JSON):
    {
      "confirmed_missing": {
        "YYYY-MM-DD": "ISO-8601 timestamp when first marked"
      }
    }

File location:  <out_root>/<SYM>/_manifest.json
                (mirrors the book-depth ingester layout)

Thread safety:  one threading.Lock per (out_root, sym) key, created lazily.
                Safe under ThreadPoolExecutor AND ProcessPoolExecutor
                (process-level fork creates independent lock namespaces --
                within-process concurrent calls are serialised; cross-process
                races land on the atomic tmp+rename, making the worst case a
                harmless overwrite by one process).

Atomic write:   tmp file + os.replace (atomic on POSIX; best-effort on Win32
                where the final rename may fail if a reader holds the file;
                we catch that and log instead of crashing).
"""
from __future__ import annotations

import datetime
import json
import os
import threading
from pathlib import Path


# Default: re-attempt a confirmed-missing date after 30 days (files occasionally
# become available after a Binance Vision republish or exchange API fix).
_DEFAULT_RECHECK_STALE_DAYS: int = 30

_MANIFEST_NAME = "_manifest.json"

# Global lock registry: maps (str(out_root), sym) -> threading.Lock
_lock_registry: dict[tuple[str, str], threading.Lock] = {}
_registry_lock = threading.Lock()


def _get_lock(out_root: Path, sym: str) -> threading.Lock:
    key = (str(out_root), sym)
    with _registry_lock:
        if key not in _lock_registry:
            _lock_registry[key] = threading.Lock()
        return _lock_registry[key]


class MissingManifest:
    """Per-asset confirmed-missing manifest.

    Parameters
    ----------
    out_root:
        Root directory that holds per-asset sub-directories.
        The manifest for ``sym`` lives at ``out_root / sym / _manifest.json``.
    recheck_stale_days:
        How many days before a confirmed-missing entry is considered stale
        and re-attempted.  Default 30 (matches book-depth ingester).
    """

    def __init__(self, out_root: Path, recheck_stale_days: int = _DEFAULT_RECHECK_STALE_DAYS):
        self._out_root = Path(out_root)
        self._recheck_stale_days = recheck_stale_days

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_known_missing(self, sym: str, date_str: str) -> bool:
        """Return True if ``date_str`` is in the manifest AND not yet stale.

        A stale entry (older than recheck_stale_days) returns False so the
        caller re-attempts the fetch (files sometimes become available later).
        """
        m = self._load(sym)
        return _is_confirmed_missing(m, date_str, self._recheck_stale_days)

    def mark_missing(self, sym: str, date_str: str) -> None:
        """Persistently record that ``date_str`` returned a 404 / empty."""
        lock = _get_lock(self._out_root, sym)
        with lock:
            m = self._load(sym)
            m["confirmed_missing"][date_str] = datetime.datetime.utcnow().isoformat()
            self._save(sym, m)

    def unmark_missing(self, sym: str, date_str: str) -> None:
        """Remove ``date_str`` from confirmed_missing (called after a successful fetch).

        No-op if the entry does not exist.
        """
        lock = _get_lock(self._out_root, sym)
        with lock:
            m = self._load(sym)
            if m.get("confirmed_missing", {}).pop(date_str, None) is not None:
                self._save(sym, m)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _manifest_path(self, sym: str) -> Path:
        return self._out_root / sym / _MANIFEST_NAME

    def _load(self, sym: str) -> dict:
        p = self._manifest_path(sym)
        if not p.exists():
            return {"confirmed_missing": {}}
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if "confirmed_missing" not in d:
                d["confirmed_missing"] = {}
            return d
        except Exception:
            return {"confirmed_missing": {}}

    def _save(self, sym: str, m: dict) -> None:
        p = self._manifest_path(sym)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: write to .tmp then rename.  On Windows, rename over an
        # existing file is atomic if both are on the same filesystem volume.
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(m, indent=2, sort_keys=True), encoding="utf-8")
        try:
            # os.replace is atomic on POSIX; on Win32 it is best-effort (may
            # fail if another process holds the file open, but that is very
            # rare for a small JSON manifest).
            os.replace(str(tmp), str(p))
        except OSError:
            # Fallback: p.replace() which is equivalent but may raise on some
            # Win32 configurations; swallow and let the next read rebuild.
            try:
                tmp.replace(p)
            except OSError as e:
                print(f"  [manifest] WARNING: could not atomic-rename {tmp} -> {p}: {e}",
                      flush=True)


# ------------------------------------------------------------------
# Module-level helpers (mirror the standalone functions in bd ingester)
# so callers that import directly can use them without instantiating the class.
# ------------------------------------------------------------------

def _is_confirmed_missing(manifest: dict, date_str: str,
                           recheck_stale_days: int = _DEFAULT_RECHECK_STALE_DAYS) -> bool:
    """True if date_str is confirmed-missing AND the mark is within recheck_stale_days."""
    entry = manifest.get("confirmed_missing", {}).get(date_str)
    if not entry:
        return False
    try:
        marked = datetime.datetime.fromisoformat(entry).date()
    except (ValueError, TypeError):
        return False
    age_days = (datetime.date.today() - marked).days
    return age_days < recheck_stale_days
