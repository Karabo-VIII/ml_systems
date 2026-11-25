"""``mlsys certify``: certify a data root and a range before they are promoted.

One command runs these steps and reports each as PASS, WARN or FAIL:

- ``code``: the git commit of this code, and whether ``pipeline/``, ``config/`` or
  ``pyproject.toml`` differ from it.
- ``status``: every stage complete over the range.
- ``reproduce``: the range is rebuilt in place from bronze with ``--force`` (bars, features,
  dataset), and every silver and gold partition of it must equal the stored file byte for
  byte; a content fingerprint tells a changed value from a bytes-only difference. The rebuild
  is refused when it would give a dollar chain a new origin or a gold month another cohort,
  because it would then rewrite data built from another start or asset list.
  ``--no-rebuild`` skips this step, as a warning.
- ``validate``: the in-pipeline gates and the independent checks, as the validate stage runs
  them (the crypto per-asset features have their own truncation proof there).
- ``causality``: the cross-sectional columns, recomputed with every input cut at sampled bars,
  equal the stored values bit for bit: crypto gold ``xd_*``, and every FX feature column (the
  FX features stage builds the currency strengths across pairs).
- ``crosscheck``: public APIs the build never read (crypto: Binance REST, funding and a second
  venue; FX: the ECB reference rates). ``--refresh`` fetches every response live;
  ``--skip-network`` skips this step, as a warning.

The certificate, ``certify/<market>_<start>_<end>.json`` under the data root, records the
commit, the library versions, the range, every check (with the validate and cross-check
details) and digests of the content and of the bytes of the range's partitions. The exit code
is 0, 1 or 2, as for ``run``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import platform
import struct
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

from pipeline.calendar import iter_days, iter_months
from pipeline.constants import CONTRACT_VERSION, DAY_MS
from pipeline.io import atomic_write_json, file_sha256
from pipeline.paths import (
    bars_month_path,
    dollar_state_meta_path,
    features_month_path,
    gold_manifest_path,
    gold_month_path,
    trades_path,
)
from pipeline.validate_run import Report

REPO = Path(__file__).resolve().parents[1]
SHOWN = 6


# --- small helpers ---------------------------------------------------------------------

def _listing(items: list[str], shown: int = SHOWN) -> str:
    more = len(items) - shown
    return ", ".join(items[:shown]) + (f" (+{more} more)" if more > 0 else "")


def _counts(report: Report) -> dict[str, int]:
    out = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for check in report.checks:
        out[check.severity] = out.get(check.severity, 0) + 1
    return out


def _severity(code: int) -> str:
    return {0: "PASS", 1: "WARN"}.get(code, "FAIL")


def _when(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- code identity ---------------------------------------------------------------------

CODE_PATHS = ("pipeline", "config", "pyproject.toml")


def code_identity(repo: Path = REPO) -> dict:
    """The commit this code is at, its branch, and the paths of the code (``pipeline/``,
    ``config/``, ``pyproject.toml``) that differ from it, untracked files included. A change
    elsewhere (a document, a scratch file) does not change what the pipeline builds."""

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout if out.returncode == 0 else None

    commit, branch = git("rev-parse", "HEAD"), git("rev-parse", "--abbrev-ref", "HEAD")
    changes = git("status", "--porcelain", "--untracked-files=normal", "--", *CODE_PATHS)
    changed = [line[3:] for line in (changes or "").splitlines() if line.strip()]
    return {
        "commit": commit.strip() if commit else None,
        "branch": branch.strip() if branch else None,
        "dirty": None if changes is None else bool(changed),
        "changed": changed,
    }


# --- the partitions of a range and their fingerprints ----------------------------------

def range_partitions(cfg, assets: list[str]) -> list[Path]:
    """The silver and gold partitions a forced bars, features and dataset run over the
    range rewrites: bars, features and gold of every cadence, and the daily panel."""
    from pipeline.chunked import _daily_path

    root, out = cfg.data_root, []
    for asset in assets:
        for y, m in iter_months(cfg.start, cfg.end):
            for bar_type in cfg.bar_types:
                for path_fn in (bars_month_path, features_month_path, gold_month_path):
                    path = path_fn(root, bar_type, asset, y, m)
                    if path.exists():
                        out.append(path)
            daily = _daily_path(root, asset, y, m)
            if daily.exists():
                out.append(daily)
    return out


def fingerprint(path: Path) -> dict:
    """The file's sha256, and a digest of its content (shape, dtypes and the row hashes in
    order) that does not depend on how the bytes are laid out."""
    df = pl.read_parquet(path)
    h = df.hash_rows(seed=7).to_numpy().astype(np.uint64)
    w = np.arange(1, len(h) + 1, dtype=np.uint64)
    shape = [df.height, df.columns, [str(t) for t in df.dtypes],
             int(h.sum(dtype=np.uint64)), int((h * w).sum(dtype=np.uint64))]
    return {"sha256": file_sha256(path), "content": hashlib.sha256(json.dumps(shape).encode()).hexdigest()[:32]}


def fingerprints(cfg, assets: list[str]) -> dict[str, dict]:
    return {p.relative_to(cfg.data_root).as_posix(): fingerprint(p) for p in range_partitions(cfg, assets)}


def _digest(fps: dict[str, dict], key: str) -> str:
    h = hashlib.sha256()
    for name in sorted(fps):
        h.update(f"{name}={fps[name][key]}\n".encode())
    return h.hexdigest()


# --- reproduction ----------------------------------------------------------------------

def expected_cohort(cfg, assets: list[str], year: int, month: int) -> list[str] | None:
    """The cohort ``write_dataset_month`` gives a month: every asset whose range reaches
    it, when there are at least two."""
    from pipeline.chunked import _asset_range

    members = []
    for asset in assets:
        start, end = _asset_range(cfg, asset)
        if (start.year, start.month) <= (year, month) <= (end.year, end.month):
            members.append(asset)
    return sorted(members) if len(members) > 1 else None


def rebuild_refusals(cfg, assets: list[str]) -> list[str]:
    """Why a forced rebuild of this range would rewrite stored data instead of reproducing it.

    A forced rebuild writes the first and the last month of the range clipped to it, restarts
    every dollar chain at the range's start and rebuilds gold with the cohort of this asset
    list. Stored bars outside the range in those months, a stored chain that calibrated on
    another day, or a gold month built with another cohort came from another range or asset
    list.
    """
    from pipeline.chunked import _asset_range

    out: list[str] = []
    for asset in assets:
        start, end = _asset_range(cfg, asset)
        if start > end:
            continue
        lo = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp() * 1000)
        hi = int(datetime(end.year, end.month, end.day, tzinfo=timezone.utc).timestamp() * 1000) + DAY_MS
        for bar_type in cfg.bar_types:
            first = bars_month_path(cfg.data_root, bar_type, asset, start.year, start.month)
            last = bars_month_path(cfg.data_root, bar_type, asset, end.year, end.month)
            before = first.exists() and (pl.read_parquet(first, columns=["ts_ms"])["ts_ms"] <= lo).any()
            after = last.exists() and (pl.read_parquet(last, columns=["ts_ms"])["ts_ms"] > hi).any()
            if before or after:
                out.append(f"{asset} has stored {bar_type} bars {'before ' + str(start) if before else 'after ' + str(end)}")
                break
    if cfg.market == "fx":
        return out
    if "dollar" in cfg.bar_types:
        for asset in assets:
            meta = dollar_state_meta_path(cfg.data_root, asset)
            if not meta.exists():
                continue
            cal = json.loads(meta.read_text()).get("calibration_day")
            if cal is None:
                continue
            start, end = _asset_range(cfg, asset)
            first = next((d for d in iter_days(start, end) if trades_path(cfg.data_root, asset, d).exists()), None)
            stored = date(1970, 1, 1) + timedelta(days=int(cal))
            if first is not None and first != stored:
                out.append(f"{asset}'s dollar chain calibrated on {stored}, a rebuild from this start calibrates on {first}")
    for y, m in iter_months(cfg.start, cfg.end):
        want = expected_cohort(cfg, assets, y, m)
        for bar_type in cfg.bar_types:
            for asset in assets:
                manifest = gold_manifest_path(gold_month_path(cfg.data_root, bar_type, asset, y, m))
                if not manifest.exists():
                    continue
                have = json.loads(manifest.read_text()).get("cohort")
                if have != want:
                    out.append(f"gold {bar_type} {y:04d}-{m:02d} was built with the cohort {have}, this asset list gives {want}")
                break  # every member of a month's cohort records the same cohort
    return out


def _reproduce(cfg, assets: list[str], report: Report) -> dict[str, dict]:
    from pipeline.orchestrator import run

    refusals = rebuild_refusals(cfg, assets)
    if refusals:
        report.add("reproduce", "FAIL",
                   "not rebuilt: a forced rebuild would rewrite data built from another start or asset list ("
                   + _listing(refusals) + "); certify with the start and assets the data was built with")
        return fingerprints(cfg, assets)
    print(f"[certify] reproduce: rebuilding {cfg.start} -> {cfg.end} in place from bronze with --force")
    before = fingerprints(cfg, assets)
    code = run(dataclasses.replace(cfg, stages=["bars", "features", "dataset"], force=True))
    after = fingerprints(cfg, assets)
    common = before.keys() & after.keys()
    changed = sorted(k for k in common if before[k]["content"] != after[k]["content"])
    bytes_only = sorted(k for k in common
                        if before[k]["content"] == after[k]["content"] and before[k]["sha256"] != after[k]["sha256"])
    gone, new = sorted(before.keys() - after.keys()), sorted(after.keys() - before.keys())
    if code >= 2:
        report.add("reproduce", "FAIL", f"the rebuild failed (exit {code}); see its log")
    elif changed or bytes_only or gone or new:
        parts = []
        for items, what in ((changed, "changed content"), (bytes_only, "kept their content but not their bytes"),
                            (gone, "disappeared"), (new, "are new")):
            if items:
                parts.append(f"{len(items)} {what} ({_listing(items)})")
        report.add("reproduce", "FAIL", f"of {len(before)} stored partitions, " + "; ".join(parts))
    else:
        report.add("reproduce", "PASS",
                   f"{len(after)} partitions rebuilt from bronze with --force are byte-identical to the stored files")
    return after


# --- validation and the public-API cross-check ------------------------------------------

def _validate(cfg, network: bool, report: Report) -> list[dict]:
    from pipeline.orchestrator import run

    reports: dict = {}
    code = run(dataclasses.replace(cfg, stages=["validate"], skip_network=not network), reports=reports)
    got = reports.get("validate")
    if got is None:
        report.add("validate", "FAIL", f"the validate stage gave no report (exit {code}); see its log")
        return []
    counts = _counts(got)
    flagged = [f"{c.name} ({c.severity})" for c in got.checks if c.severity != "PASS"]
    detail = f"PASS={counts['PASS']} WARN={counts['WARN']} FAIL={counts['FAIL']}"
    report.add("validate", _severity(max(code, got.code)), detail + (f"; {_listing(flagged)}" if flagged else ""))
    return [dataclasses.asdict(c) for c in got.checks]


def _crosscheck(cfg, assets: list[str], refresh: bool, report: Report) -> list[dict]:
    from pipeline.crosscheck import crosscheck

    got = crosscheck(cfg.data_root, assets, cfg.start, cfg.end,
                     None if cfg.market == "fx" else list(cfg.bar_types), refresh=refresh, market=cfg.market)
    print()
    print("=== public-API cross-check ===")
    got.print()
    counts = _counts(got)
    flagged = [f"{c.name} ({c.severity})" for c in got.checks if c.severity != "PASS"]
    how = "every response fetched live" if refresh else "cached responses where present (--refresh fetches them live)"
    detail = f"PASS={counts['PASS']} WARN={counts['WARN']} FAIL={counts['FAIL']}, {how}"
    report.add("crosscheck", _severity(got.code), detail + (f"; {_listing(flagged)}" if flagged else ""))
    return [dataclasses.asdict(c) for c in got.checks]


# --- causality: truncation proofs of the cross-sectional columns -------------------------

def proof_months(cfg, assets: list[str]) -> list[tuple[int, int]]:
    """The first and the last month of the range, and every month in which an asset starts
    (a listing changes the cross-section in the middle of a month)."""
    from pipeline.chunked import _asset_range

    months = {(cfg.start.year, cfg.start.month), (cfg.end.year, cfg.end.month)}
    for asset in assets:
        start, _end = _asset_range(cfg, asset)
        if cfg.start < start <= cfg.end:
            months.add((start.year, start.month))
    return sorted(months)


def _same(a, b) -> bool:
    """Equal bit for bit. A null and a NaN both mean no value."""
    missing_a = a is None or (isinstance(a, float) and math.isnan(a))
    missing_b = b is None or (isinstance(b, float) and math.isnan(b))
    if missing_a or missing_b:
        return missing_a and missing_b
    if isinstance(a, float) and isinstance(b, float):
        return struct.pack("<d", a) == struct.pack("<d", b)
    return a == b


def truncation_proof(cfg, bar_type: str, months: list[tuple[int, int]], *, members_for, stored_path,
                     build, columns) -> tuple[str, str]:
    """Recompute months with every input cut at two bars (the middle one and the last one)
    and compare each stored row at the cut, column by column, bit for bit.

    ``build(cfg, members, bar_type, year, month, cut_ms=..., sink=...)`` is the stage function,
    ``members_for(year, month)`` the cohort it is given, ``stored_path`` the partition path
    function and ``columns(frame)`` the columns compared. Returns a severity and a detail.
    """
    compared = n_cuts = 0
    any_columns = False
    bad: list[str] = []
    seen: list[str] = []
    for y, m in months:
        members = members_for(y, m)
        stored: dict[str, pl.DataFrame] = {}
        for asset in members:
            path = stored_path(cfg.data_root, bar_type, asset, y, m)
            if path.exists():
                frame = pl.read_parquet(path).sort("ts_ms", maintain_order=True)
                if frame.height:
                    stored[asset] = frame
        if not stored:
            continue
        seen.append(f"{y:04d}-{m:02d}")
        ts = stored[min(stored)]["ts_ms"]
        for cut in sorted({int(ts[ts.len() // 2]), int(ts[-1])}):
            got: dict[str, pl.DataFrame] = {}
            build(cfg, members, bar_type, y, m, cut_ms=cut, sink=lambda a, f: got.__setitem__(a, f))
            n_cuts += 1
            for asset, frame in stored.items():
                cols = columns(frame)
                any_columns = any_columns or bool(cols)
                want = frame.filter(pl.col("ts_ms") == cut)
                if want.height != 1 or not cols:
                    continue
                have = got.get(asset)
                have = None if have is None else have.filter(pl.col("ts_ms") == cut)
                if have is None or have.height != 1:
                    bad.append(f"{asset} {_when(cut)}: no row when the inputs are cut there")
                    continue
                w, g = want.row(0, named=True), have.row(0, named=True)
                for c in cols:
                    compared += 1
                    if not _same(w.get(c), g.get(c)):
                        bad.append(f"{asset} {c} at {_when(cut)}: stored {w.get(c)!r}, cut {g.get(c)!r}")
    where = f"{n_cuts} cut(s) in {', '.join(seen) or 'no month'}"
    if bad:
        return "FAIL", f"{len(bad)} value(s) change when the inputs are cut at the bar ({where}): {_listing(bad)}"
    if not any_columns:
        return "PASS", f"no cross-sectional columns to prove ({where}): one asset per month"
    if compared == 0:
        return "FAIL", f"compared no value ({where}); the proof did not run"
    return "PASS", f"{compared} values at {where} equal a recompute from inputs cut at the bar, bit for bit"


def _causality(cfg, assets: list[str], months: list[tuple[int, int]], report: Report) -> None:
    from pipeline.chunked import write_dataset_month, write_fx_features_month

    for bar_type in cfg.bar_types:
        if cfg.market == "fx":
            severity, detail = truncation_proof(
                cfg, bar_type, months,
                members_for=lambda y, m: list(assets),
                stored_path=features_month_path,
                build=lambda c, members, bt, y, m, cut_ms, sink: write_fx_features_month(
                    c, members, bt, y, m, cut_ms=cut_ms, sink=sink),
                columns=lambda frame: [c for c in frame.columns if c != "ts_ms"],
            )
            name = f"causality:fx_features:{bar_type}"
        else:
            def members_for(y: int, m: int, bt: str = bar_type) -> list[str]:
                # The cohort the stored month was built with, so that the proof recomputes
                # the same cross-section even when this asset list differs from the build's.
                for asset in assets:
                    manifest = gold_manifest_path(gold_month_path(cfg.data_root, bt, asset, y, m))
                    if manifest.exists():
                        cohort = json.loads(manifest.read_text()).get("cohort")
                        return list(cohort) if cohort else [asset]
                return list(assets)

            severity, detail = truncation_proof(
                cfg, bar_type, months,
                members_for=members_for,
                stored_path=gold_month_path,
                build=lambda c, members, bt, y, m, cut_ms, sink: write_dataset_month(
                    c, members, bt, y, m, {}, cut_ms=cut_ms, sink=sink),
                columns=lambda frame: [c for c in frame.columns if c.startswith("xd_")],
            )
            name = f"causality:gold_xd:{bar_type}"
        report.add(name, severity, detail)


# --- the command -----------------------------------------------------------------------

def certificate_path(cfg) -> Path:
    return cfg.data_root / "certify" / f"{cfg.market}_{cfg.start.isoformat()}_{cfg.end.isoformat()}.json"


def certify(cfg, *, rebuild: bool = True, network: bool = True, refresh: bool = False) -> Report:
    """Certify ``cfg``'s range and assets; see the module docstring. Returns the report."""
    import pyarrow

    from pipeline.config import load_splits, load_windows
    from pipeline.orchestrator import resolve_assets, status

    cfg = dataclasses.replace(cfg, windows=cfg.windows or load_windows(), splits=cfg.splits or load_splits(),
                              force=False, dry_run=False)
    assets = resolve_assets(cfg)
    started = _now()
    report = Report()

    code = code_identity()
    if code["commit"] is None:
        report.add("code", "WARN", "not a git checkout, or git is unavailable: the certificate cannot name the code")
    elif code["dirty"]:
        report.add("code", "WARN", f"commit {code['commit'][:12]} on {code['branch']} with uncommitted changes: "
                                   + _listing(code["changed"]))
    else:
        report.add("code", "PASS", f"commit {code['commit'][:12]} on {code['branch']}, no uncommitted changes")
    print(f"[certify] market={cfg.market} {cfg.start} -> {cfg.end} assets={','.join(assets)} "
          f"root={cfg.data_root} commit={(code['commit'] or 'unknown')[:12]}")

    complete = status(cfg=dataclasses.replace(cfg, stages=["fetch", "bars", "features", "dataset"])) == 0
    report.add("status", "PASS" if complete else "FAIL",
               "every stage is complete over the range" if complete else "incomplete; see the status table above")

    fps = None
    if not rebuild:
        report.add("reproduce", "WARN", "not run (--no-rebuild): the stored data is not shown to be what this code builds")
    elif not complete:
        report.add("reproduce", "WARN", "not run: the range is incomplete")
    else:
        fps = _reproduce(cfg, assets, report)

    details = {"validate": _validate(cfg, network, report)}
    _causality(cfg, assets, proof_months(cfg, assets), report)
    if network:
        details["crosscheck"] = _crosscheck(cfg, assets, refresh, report)
    else:
        report.add("crosscheck", "WARN", "not run (--skip-network)")

    if fps is None:
        fps = fingerprints(cfg, assets)
    certificate = {
        "certificate": "mlsys certify",
        "contract_version": CONTRACT_VERSION,
        "code": code,
        "environment": {
            "python": platform.python_version(), "polars": pl.__version__, "pyarrow": pyarrow.__version__,
            "numpy": np.__version__, "platform": platform.platform(),
        },
        "market": cfg.market,
        "range": [cfg.start.isoformat(), cfg.end.isoformat()],
        "assets": assets,
        "bar_types": list(cfg.bar_types),
        "options": {"rebuild": rebuild, "network": network, "refresh": refresh},
        "started_at": started,
        "finished_at": _now(),
        "checks": [dataclasses.asdict(c) for c in report.checks],
        "details": details,
        "data": {"partitions": len(fps), "content_digest": _digest(fps, "content"), "sha256_digest": _digest(fps, "sha256")},
        "summary": _counts(report),
        "exit": report.code,
    }
    path = certificate_path(cfg)
    atomic_write_json(path, certificate)
    print()
    print("=== certification ===")
    report.print()
    print(f"certificate: {path}")
    return report
