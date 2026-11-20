"""Shared pipeline progress + logging helpers.

Purpose
=======
Single source of truth for what producer scripts emit. Closes the
homogeneous-interface gap surfaced by oracle 2026-05-22:

  - `fetch_all.py` had GOOD progress: tqdm bars + `[BULK API]`, `[OK]`,
    `[SKIP]`, `[WARN]`, `[DL]` prefixes.
  - `refresh.py` only emitted heartbeats (terminal silent until 30s tick).
  - Producers like `basis_signals.py` had ad-hoc prints with varying
    conventions (`[basis_feat scan i/N]` etc).
  - When run THROUGH refresh.py, producer stdout was captured to a log
    file and never echoed to terminal — so even the good producers were
    silent to the user.

This module provides a SHARED vocabulary so:
  1. Every producer's terminal output follows the same prefix format.
  2. The output is parseable (refresh.py's heartbeat tail picks up the
     latest state cleanly; CI tools can grep `[OK]` / `[FAIL]`).
  3. The same progress works when the producer runs standalone OR via
     refresh.py streaming (commit refactor 2026-05-22).

Convention
==========
Every producer line follows this shape:

    [<module>] [<phase>] <message> [optional: i/N elapsed=Xs eta=Ys]

Phase markers (canonical set; use exactly these tokens):

    START    | begin of a stage
    SCAN     | enumerate files / discover work
    DL       | download / fetch
    PARSE    | parse / decode raw bytes
    BUILD    | compute / aggregate
    WRITE    | atomic_write_parquet / save
    GATE     | validate / sanity check
    OK       | step / phase completed cleanly
    SKIP     | step intentionally skipped (with reason)
    WARN     | non-fatal anomaly
    FAIL     | hard error (next: log details, propagate exit code)
    DONE     | producer complete (always last line)

Usage
=====
Producer side:

    from pipeline.progress import phase_log, ProgressTask

    phase_log("basis_feat", "START", "rebuild basis_features_long")
    with ProgressTask("basis_feat", total=len(files), label="scan") as bar:
        for f in files:
            do_work(f)
            bar.update(1)
    phase_log("basis_feat", "OK", f"wrote {n_rows} rows")
    phase_log("basis_feat", "DONE", "elapsed=42s")

Standalone: tqdm bar + line prints, autoflush.
Through refresh.py: same lines stream to terminal, captured to log
file simultaneously (refresh.py's tee thread, commit 2026-05-22).

Design notes
============
- No global state. Each call computes its own prefix.
- tqdm is optional — if not installed or stdout is not a TTY, falls back
  to incremental [i/N] prints every PROGRESS_PRINT_EVERY=10% or every
  PROGRESS_PRINT_SECONDS=30s, whichever comes first.
- `phase_log` is fast (single print, no formatting work for ok-path lines).
- ANSI colors deliberately NOT used — refresh.py logs are downstream
  parsed by CDAP and the live dashboard; color codes pollute grep.
"""
from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from typing import Iterable, Iterator, Optional

# CDAP contract
__contract__ = {
    "kind": "pipeline_helper",
    "stage": "progress_formatter",
    "inputs": {
        "module": "producer name (e.g., 'basis_feat', 'chimera_v51')",
        "phase": "one of START | SCAN | DL | PARSE | BUILD | WRITE | GATE | OK | SKIP | WARN | FAIL | DONE",
    },
    "outputs": {
        "stdout_lines": "[<module>] [<phase>] <message> ...",
    },
    "invariants": {
        "no_network": True,
        "no_data_write": True,
        "deterministic_format": True,
        "no_ansi_color": True,
    },
    "rationale": "Homogeneous producer progress + heartbeat-compatible. Closes pipeline progress gap surfaced by oracle 2026-05-22.",
}

VALID_PHASES = frozenset([
    "START", "SCAN", "DL", "PARSE", "BUILD", "WRITE", "GATE",
    "OK", "SKIP", "WARN", "FAIL", "DONE",
])

PROGRESS_PRINT_EVERY: float = 0.10  # print at every 10% increment (fallback)
PROGRESS_PRINT_SECONDS: float = 30.0  # or every 30 seconds, whichever first

# Try to import tqdm; if absent, use the fallback printer.
try:
    from tqdm import tqdm as _tqdm  # type: ignore
    _HAS_TQDM = True
except ImportError:  # pragma: no cover
    _tqdm = None
    _HAS_TQDM = False


def phase_log(module: str, phase: str, message: str, *,
              counters: Optional[dict] = None,
              flush: bool = True) -> None:
    """Emit a standardized progress line.

    Args:
        module:   short producer name, lower_snake (e.g. 'basis_feat', 'chimera_v51')
        phase:    one of VALID_PHASES (raises ValueError otherwise)
        message:  free-form human-readable message
        counters: optional dict like {"i": 17, "N": 92, "elapsed": 42.3}.
                  Will be appended as `[i/N elapsed=Xs]` (only for i/N or elapsed).
        flush:    flush stdout after print (default True; set False inside
                  a tqdm bar to avoid line interleaving with the bar refresh).

    Returns: None. Prints to stdout. Never raises (except for invalid phase).

    Example:
        phase_log("basis_feat", "OK", "wrote 1.2M rows",
                  counters={"i": 92, "N": 92, "elapsed": 87.4})
        → "[basis_feat] [OK] wrote 1.2M rows  [92/92 elapsed=87s]"
    """
    if phase not in VALID_PHASES:
        raise ValueError(
            f"phase={phase!r} not in {sorted(VALID_PHASES)}. "
            f"Pick the closest token; if none fits, propose a new one."
        )
    line = f"[{module}] [{phase}] {message}"
    if counters:
        bits = []
        if "i" in counters and "N" in counters:
            bits.append(f"{counters['i']}/{counters['N']}")
        if "elapsed" in counters:
            bits.append(f"elapsed={counters['elapsed']:.0f}s")
        if "eta" in counters:
            bits.append(f"eta={counters['eta']:.0f}s")
        if bits:
            line = f"{line}  [{' '.join(bits)}]"
    print(line, flush=flush)


class ProgressTask:
    """Context manager for a counted-progress block.

    Uses tqdm when available + stdout is a TTY; falls back to incremental
    `[i/N elapsed=Xs]` prints at coarser intervals otherwise.

    Use this when you have a known total (N) and want a visual bar OR
    well-spaced terminal updates.

    Example:
        with ProgressTask("chimera_v51", total=len(assets), label="build") as bar:
            for asset in assets:
                build_one(asset)
                bar.update(1)

    Properties:
        - Always emits a START line at __enter__
        - Always emits a DONE line at __exit__ (with elapsed)
        - bar.update(1) advances the counter; under fallback mode, only
          prints when crossing PROGRESS_PRINT_EVERY threshold or
          PROGRESS_PRINT_SECONDS interval
        - bar.set_label("scan") changes the in-flight label
    """

    def __init__(self, module: str, total: int, label: str = "task",
                 force_fallback: bool = False) -> None:
        self.module = module
        self.total = max(0, int(total))
        self.label = label
        self.i = 0
        self._t0 = 0.0
        self._tqdm_bar = None
        self._last_print_pct = -1.0
        self._last_print_t = 0.0
        # Decide tqdm vs fallback once at start
        self._use_tqdm = (
            _HAS_TQDM and not force_fallback
            and sys.stdout.isatty()
            and os.environ.get("PIPELINE_PROGRESS_NO_TQDM") != "1"
        )

    def __enter__(self) -> "ProgressTask":
        self._t0 = time.time()
        phase_log(self.module, "START", f"{self.label}: total={self.total}")
        if self._use_tqdm and self.total > 0:
            self._tqdm_bar = _tqdm(total=self.total, desc=f"[{self.module}] {self.label}",
                                    leave=False, ncols=88, mininterval=0.5)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._tqdm_bar is not None:
            try:
                self._tqdm_bar.close()
            except Exception:
                pass
        elapsed = time.time() - self._t0
        if exc_type is None:
            phase_log(self.module, "OK",
                      f"{self.label} complete: {self.i}/{self.total}",
                      counters={"elapsed": elapsed})
        else:
            phase_log(self.module, "FAIL",
                      f"{self.label} aborted at {self.i}/{self.total}: {exc_type.__name__}",
                      counters={"elapsed": elapsed})

    def update(self, n: int = 1, *, msg: Optional[str] = None) -> None:
        """Advance counter by n. Optionally emit an inline message."""
        self.i += n
        if self._tqdm_bar is not None:
            try:
                self._tqdm_bar.update(n)
                if msg is not None:
                    self._tqdm_bar.set_postfix_str(msg, refresh=False)
            except Exception:
                pass
            return
        # Fallback path: print at thresholds
        now = time.time()
        pct = (self.i / self.total) if self.total > 0 else 1.0
        crossed_pct = pct - self._last_print_pct >= PROGRESS_PRINT_EVERY
        crossed_time = (now - self._last_print_t) >= PROGRESS_PRINT_SECONDS
        if self.i == 1 or crossed_pct or crossed_time or self.i == self.total:
            self._last_print_pct = pct
            self._last_print_t = now
            elapsed = now - self._t0
            eta = (elapsed / pct - elapsed) if pct > 0 else 0
            extra = f" — {msg}" if msg else ""
            phase_log(self.module, "BUILD", f"{self.label}{extra}",
                      counters={"i": self.i, "N": self.total,
                                "elapsed": elapsed, "eta": eta})

    def set_label(self, label: str) -> None:
        """Change the in-flight label (e.g., 'scan' → 'parse' → 'write')."""
        self.label = label
        if self._tqdm_bar is not None:
            try:
                self._tqdm_bar.set_description(f"[{self.module}] {label}")
            except Exception:
                pass


@contextmanager
def stage_run(module: str, description: str = "") -> Iterator[None]:
    """Context manager wrapping a producer's whole run.

    Emits START + DONE + elapsed automatically. Catches exceptions and
    re-raises them after emitting FAIL.

    Example:
        with stage_run("basis_feat", "rebuild basis_features_long"):
            main_work()
    """
    t0 = time.time()
    phase_log(module, "START", description or "stage begin")
    try:
        yield
    except SystemExit as e:
        # A clean sys.exit(0) is success, not a failure -- don't emit FAIL (which
        # the refresh.py log parser would read as a failed stage). Only non-zero
        # exits are failures.
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        if code == 0:
            phase_log(module, "DONE", description or "stage end",
                      counters={"elapsed": time.time() - t0})
        else:
            phase_log(module, "FAIL", f"SystemExit({code})",
                      counters={"elapsed": time.time() - t0})
        raise
    except BaseException as e:
        phase_log(module, "FAIL", f"{type(e).__name__}: {e}",
                  counters={"elapsed": time.time() - t0})
        raise
    phase_log(module, "DONE", description or "stage end",
              counters={"elapsed": time.time() - t0})


def progress_iter(it: Iterable, module: str, total: Optional[int] = None,
                  label: str = "iter") -> Iterator:
    """Wrap an iterable to emit ProgressTask updates.

    Example:
        for f in progress_iter(files, "basis_feat", total=len(files), label="scan"):
            do_work(f)
    """
    if total is None:
        try:
            total = len(it)  # type: ignore
        except Exception:
            total = 0
    with ProgressTask(module, total or 0, label=label) as bar:
        for item in it:
            yield item
            bar.update(1)


__all__ = [
    "phase_log", "ProgressTask", "stage_run", "progress_iter",
    "VALID_PHASES", "PROGRESS_PRINT_EVERY", "PROGRESS_PRINT_SECONDS",
]
