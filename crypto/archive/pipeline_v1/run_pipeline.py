"""End-to-end pipeline DAG runner with per-stage usability tracking.

Single entry point for refreshing the data pipeline from raw → gold. Each
stage produces concrete artifacts that downstream consumers can already use.

Stage map (each stage has clear "what's usable after it completes"):

  T0  fetch_binance        raw aggTrades / funding / metrics              -> data/raw/
                           ✓ usable for: bar_fabric, dollar bar generation,
                                          any tick-level analysis
  T0  fetch_external       farside / defillama / deribit / etc            -> data/raw_external/
                           ✓ usable for: stable-flow / ETF / DVOL / wiki
                                          overlays directly

  T1  bar_fabric           DIB / runs_tick / runs_volume / range          -> data/processed/bars/
                           ✓ usable for: bar-type-specific strategies (DIB
                                          flow duo, runs scalping, volatility
                                          range breakouts)
  T1  hawkes_branching     Hawkes branching ratio panel                    -> data/processed/hawkes/daily/
                           ✓ usable for: ranker feature, microstructure
                                          analysis

  T2  build_panels         S3 / basis / liquidations / whale (ingestor    -> data/processed/panels/daily/
                           feature scripts)
                           ✓ usable for: cross-asset overlays directly,
                                          parity audits

  T3  frontier_consolidate per-asset 80-feature daily silver               -> data/processed/frontier/daily/
                           ✓ usable for: ml_training_data extraction,
                                          ranker training, daily-cadence
                                          strategies

  T4  chimera_legacy       v50 chimera (41 features)                      -> data/processed/chimera_legacy/dollar/
                           ✓ usable for: V1-V14 inference + retraining,
                                          legacy paper trader

  T5  chimera_v51          v51 chimera (154 cols) + 4 cadence views       -> data/processed/chimera/{dollar,1d,4h,1h,15m}/
                           + manifest                                       data/manifests/v51_<SYM>.json
                           ✓ usable for: V0/V1.x f121 training,
                                          ChimeraLoader, full strategy stack

  T6  validate             validate_chimera + cross_asset + e2e           -> logs/validate_*
                           ✓ usable for: pre-train CI gate

Usage:
    python src/pipeline/run_pipeline.py --tiers all                 # full pipeline
    python src/pipeline/run_pipeline.py --tiers fetch,bars,chimera  # subset
    python src/pipeline/run_pipeline.py --asset BTC                  # single asset
    python src/pipeline/run_pipeline.py --universe u10               # universe
    python src/pipeline/run_pipeline.py --workers 24                 # parallelism (fetch)
    python src/pipeline/run_pipeline.py --dry-run                    # preview
    python src/pipeline/run_pipeline.py --status                     # show what's usable now

Each stage logs to logs/pipeline/<stage>_<timestamp>.log and updates
data/manifests/_pipeline_state.json with last-run timestamps + status.
"""
from __future__ import annotations

# CDAP contract — declared after __future__ per PEP-236.
__contract__ = {
    "kind": "pipeline_orchestrator",
    "stage": "run_pipeline",
    "inputs": {
        "args": ["--tiers", "--asset", "--universe {u10|u50|u100}", "--workers",
                 "--stage-workers KEY=N", "--dry-run", "--status",
                 "--continue-on-fail", "--force", "--heartbeat-seconds", "--no-gc"],
    },
    "outputs": {
        "logs": "logs/pipeline/<stage>_<timestamp>.log",
        "state": "data/_manifests/_pipeline_state.json",
    },
    "invariants": {
        "chimera_legacy_runs_after_fetch_only": True,
        "chimera_v51_depends_on_v50_and_frontier": True,
        "parse_tiers_chimera_legacy_position": 1,
        "expected_output_at_stage_start": True,
        "per_asset_progress_in_heartbeat": True,
    },
}

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON = sys.executable

sys.path.insert(0, str(PROJECT_ROOT / "src" / "pipeline"))
import layout as _layout  # noqa: E402

PIPELINE_STATE_PATH = _layout.DIR_MANIFESTS / "_pipeline_state.json"
LOG_DIR = PROJECT_ROOT / "logs" / "pipeline"


# ─── Stage definitions ───────────────────────────────────────────────────────

@dataclass
class Stage:
    """One stage in the pipeline DAG."""
    key: str
    tier: int
    description: str
    usable_for: str
    cmd: list[str]
    output_check: callable = None        # () -> dict {ok, n_files, latest_mtime}
    is_stale: callable = None            # () -> bool, True = needs run
    depends_on: list[str] = field(default_factory=list)


# ─── Per-asset progress regexes ────────────────────────────────────────────
#
# Each stage's underlying script prints per-asset progress lines using
# stage-specific patterns. The heartbeat parses these to surface a rollup
# (N done / M total · running / errors) rather than just the most recent
# log line.
#
# Patterns are applied in priority order:
#   ok_re      — line matching means asset finished successfully
#   err_re     — line matching means asset failed
#   running_re — line matching means asset just started or is in-progress
#
# Add a stage by extending PROGRESS_PATTERNS. Keep regexes anchored so a
# generic OK doesn't match a different message.
import re

PROGRESS_PATTERNS: dict = {
    # bar_fabric: build_bars.py prints "  [SYM] OK ..."
    "bar_fabric": {
        "ok_re":      re.compile(r"^\s*\[([A-Z0-9]{2,12})USDT?\]\s+OK\b", re.M),
        "err_re":     re.compile(r"^\s*\[([A-Z0-9]{2,12})USDT?\]\s+(FAIL|error|EXCEPTION)\b", re.M),
        "running_re": re.compile(r"^\s*\[([A-Z0-9]{2,12})USDT?\]\s+(START|building|fetching)\b", re.M),
    },
    # hawkes_branching: "  [BTC] done: N new valid days"
    "hawkes_branching": {
        "ok_re":      re.compile(r"^\s*\[([A-Z0-9]{2,12})\]\s+done:", re.M),
        "err_re":     re.compile(r"^\s*\[([A-Z0-9]{2,12})\]\s+(worker error|FAIL)", re.M),
        "running_re": re.compile(r"^\s*\[([A-Z0-9]{2,12})\]\s+(skipping|processing|\d+/\d+ processed)", re.M),
    },
    # frontier_consolidate: "  [BTC] OK 1234r 80 features"
    "frontier_consolidate": {
        "ok_re":      re.compile(r"^\s*\[([A-Z0-9]{2,12})\]\s+OK\s+\d+r", re.M),
        "err_re":     re.compile(r"^\s*\[([A-Z0-9]{2,12})\]\s+(error|no_silver)", re.M),
        "running_re": re.compile(r"^\s*\[([A-Z0-9]{2,12})\]\s+wrote", re.M),
    },
    # chimera_legacy: tqdm + "[Phase X / SYM]" prints; capture per-asset OKs
    "chimera_legacy": {
        "ok_re":      re.compile(r"\[([A-Z0-9]{2,12})USDT?\]\s+(?:phase[12]\s+)?(?:wrote|OK|saved)", re.M | re.I),
        "err_re":     re.compile(r"\[([A-Z0-9]{2,12})USDT?\]\s+(error|FAIL|EXCEPTION)", re.M),
        "running_re": re.compile(r"\[([A-Z0-9]{2,12})USDT?\]\s+(processing|computing)", re.M | re.I),
    },
    # chimera_v51: "[BTCUSDT] OK ... v51_cols=N" or per-asset summary lines
    "chimera_v51": {
        "ok_re":      re.compile(r"\[([A-Z0-9]{2,12})USDT?\]\s+(OK|ok)\b", re.M),
        "err_re":     re.compile(r"\[([A-Z0-9]{2,12})USDT?\]\s+(error|FAIL)", re.M),
        "running_re": re.compile(r"\[([A-Z0-9]{2,12})USDT?\]\s+(BUILD|building|cadence)", re.M | re.I),
    },
    # build_panels: "[Panel N/7] <name>" for panel-level; not per-asset
    "build_panels": {
        "ok_re":      re.compile(r"^\s*\[OK\]\s+(\w+):", re.M),
        "err_re":     re.compile(r"^\s*\[FAIL\]\s+(\w+):", re.M),
        "running_re": re.compile(r"^\s*\[START\]\s+(\w+):", re.M),
    },
    # fetch_binance: handled separately (uses tqdm; per-asset/per-day)
    "fetch_binance": None,
}


def parse_per_asset_progress(stage_key: str, log_path: Path) -> dict:
    """Parse the stage's log file for per-asset progress.

    Returns dict with keys:
      done   : list[str] — assets completed OK
      err    : list[str] — assets that errored
      running: list[str] — assets currently in-flight (started but not done)
      n_done : int
      n_err  : int
      n_running : int
    Empty dict if patterns unavailable or log missing.
    """
    pats = PROGRESS_PATTERNS.get(stage_key)
    if pats is None or not log_path or not log_path.exists():
        return {}
    try:
        # Read only last ~256KB to avoid huge files
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 262144))
            text = f.read().decode("utf-8", errors="replace")
    except Exception:
        return {}

    done_set: set = set(m.group(1).upper() for m in pats["ok_re"].finditer(text)) if pats.get("ok_re") else set()
    err_set: set = set(m.group(1).upper() for m in pats["err_re"].finditer(text)) if pats.get("err_re") else set()
    run_set: set = set(m.group(1).upper() for m in pats["running_re"].finditer(text)) if pats.get("running_re") else set()
    # In-flight = started but not in done/err
    running = sorted(run_set - done_set - err_set)
    done = sorted(done_set)
    err = sorted(err_set)
    return {
        "done": done, "err": err, "running": running,
        "n_done": len(done), "n_err": len(err), "n_running": len(running),
    }


def _check_dir_files(directory: Path, glob: str = "*.parquet") -> dict:
    """Generic output check: count files + report latest mtime."""
    if not directory.exists():
        return {"ok": False, "n_files": 0, "latest_mtime": None,
                "latest_mtime_epoch": 0.0, "dir": str(directory)}
    files = list(directory.glob(glob))
    if not files:
        return {"ok": False, "n_files": 0, "latest_mtime": None,
                "latest_mtime_epoch": 0.0, "dir": str(directory)}
    latest = max(f.stat().st_mtime for f in files)
    return {
        "ok": True,
        "n_files": len(files),
        "latest_mtime": datetime.fromtimestamp(latest, tz=timezone.utc).isoformat(),
        "latest_mtime_epoch": latest,
        "dir": str(directory),
    }


def _is_stale_relative(this_dir: Path, *upstream_dirs: Path,
                       this_glob: str = "*.parquet",
                       upstream_glob: str = "*.parquet",
                       grace_seconds: int = 60) -> tuple[bool, str]:
    """Stale if any upstream file is newer than newest output in this_dir.

    Returns (is_stale, reason).
    """
    out = _check_dir_files(this_dir, this_glob)
    if not out["ok"]:
        return True, f"no outputs in {this_dir.name}"
    this_latest = out["latest_mtime_epoch"]

    for upstream in upstream_dirs:
        u = _check_dir_files(upstream, upstream_glob)
        if not u["ok"]:
            continue  # missing upstream isn't a freshness signal
        if u["latest_mtime_epoch"] > this_latest + grace_seconds:
            return True, (f"{upstream.name} is newer "
                          f"({u['latest_mtime']} vs {out['latest_mtime']})")
    return False, f"up-to-date (newest output: {out['latest_mtime']})"


def _is_stale_age(this_dir: Path, max_age_hours: float,
                  this_glob: str = "*.parquet") -> tuple[bool, str]:
    """Stale if newest output in this_dir is older than max_age_hours."""
    out = _check_dir_files(this_dir, this_glob)
    if not out["ok"]:
        return True, f"no outputs in {this_dir.name}"
    age_h = (time.time() - out["latest_mtime_epoch"]) / 3600.0
    if age_h > max_age_hours:
        return True, f"newest output is {age_h:.1f}h old (limit {max_age_hours}h)"
    return False, f"fresh ({age_h:.1f}h old)"


def _stages(asset: str | None, universe: str | None, workers: int,
            stage_workers: dict | None = None,
            force: bool = False) -> list[Stage]:
    """Build the stage list. asset/universe are passed through to scripts.

    Per-stage worker counts (`stage_workers`) override the default `workers`
    arg per stage. Fast non-polars stages get higher defaults; polars-heavy
    stages get lower (Windows segfault risk). Pass {} or None to use defaults.

    If `force=True`, every stage cmd receives `--force`, signalling each
    sub-script to delete prior dated snapshots before rebuild (uniform
    contract; bar-type sub-scripts accept --force as a no-op for uniformity).
    """
    asset_arg = ["--asset", asset] if asset else []
    asset_filter_msg = f"asset={asset}" if asset else (f"universe={universe}" if universe else "all")

    raw_dir = PROJECT_ROOT / "data" / "raw"
    sw = stage_workers or {}
    # Per-stage defaults — tuned for stability on Windows
    fetch_w = sw.get("fetch", workers)         # threaded HTTP, scales freely
    bars_w = sw.get("bars", 4)                 # concurrent BARTYPES (cap 5); each
                                                # bartype iterates assets serially.
                                                # 4 fits a 5-bartype suite without one
                                                # bartype starving the pool.
    hawkes_w = sw.get("hawkes", 8)             # per-asset, I/O-bound; 8 saturates 8-core machines
    frontier_w = sw.get("frontier", 8)         # per-asset, no big polars frames; lift from 4
    chimera_legacy_w = sw.get("chimera_legacy", 1)  # heavy polars, OOM risk
    chimera_v51_w = sw.get("chimera_v51", 1)        # heavy polars, segfault risk on Windows

    return [
        Stage(
            key="fetch_binance",
            tier=0,
            description=f"Fetch Binance aggTrades + funding + metrics ({asset_filter_msg})",
            usable_for="bar_fabric / dollar bars / tick-level analysis",
            cmd=[PYTHON, "src/pipeline/fetch_all.py", "--workers", str(fetch_w)] +
                (["--assets", f"{asset}/USDT"] if asset else []) +
                (["--universe", universe] if universe and not asset else []) +
                (["--force"] if force else []),
            output_check=lambda: _check_dir_files(raw_dir, "**/aggTrades/*.parquet"),
            # Stale if no outputs OR newest aggTrades file is >24h old
            is_stale=lambda: _is_stale_age(raw_dir, max_age_hours=24,
                                            this_glob="**/aggTrades/*.parquet"),
        ),
        Stage(
            key="bar_fabric",
            tier=1,
            description=f"Build bar fabric (DIB / runs / range / adaptive_vol) [{asset_filter_msg}, parallel_bartypes={bars_w}]",
            usable_for="bar-type-specific strategies (DIB flow duo, runs scalping)",
            cmd=[PYTHON, "src/pipeline/build_bars.py", "--workers", str(bars_w)] +
                (["--asset", asset] if asset else []) +
                (["--universe", universe] if universe and not asset else []) +
                (["--force"] if force else []),
            output_check=lambda: _check_dir_files(_layout.DIR_BARS, "**/*.parquet"),
            is_stale=lambda: _is_stale_relative(_layout.DIR_BARS, raw_dir,
                                                 this_glob="**/*.parquet",
                                                 upstream_glob="**/aggTrades/*.parquet"),
            depends_on=["fetch_binance"],
        ),
        Stage(
            key="build_panels",
            tier=2,
            description="Build multi-asset panels (s3 / basis / liquidations / whale / top_trader / etf)",
            usable_for="frontier_consolidator inputs; v51 chimera frontier features",
            cmd=[PYTHON, "src/pipeline/build_panels.py", "--skip-existing"] +
                (["--universe", universe] if universe else []) +
                (["--force"] if force else []),
            output_check=lambda: _check_dir_files(_layout.panels_dir(), "*.parquet"),
            is_stale=lambda: _is_stale_age(_layout.panels_dir(), max_age_hours=24 * 7),
            depends_on=["fetch_binance"],
        ),
        Stage(
            key="hawkes_branching",
            tier=1,
            description=f"Build Hawkes branching ratio panel (730d, parallel={hawkes_w}, universe={universe or 'u10-default'})",
            usable_for="ranker feature, microstructure analysis",
            cmd=[PYTHON, "src/pipeline/features/hawkes_branching_ratio.py",
                 "--max-days", "730", "--workers", str(hawkes_w)] +
                (["--universe", universe] if universe else []) +
                (["--assets", asset] if asset else []) +
                (["--force"] if force else []),
            output_check=lambda: _check_dir_files(_layout.hawkes_dir(), "*.parquet"),
            is_stale=lambda: _is_stale_relative(_layout.hawkes_dir(), raw_dir,
                                                 upstream_glob="**/aggTrades/*.parquet"),
            depends_on=["fetch_binance"],
        ),
        Stage(
            key="frontier_consolidate",
            tier=3,
            description=f"Consolidate per-asset frontier silver [{asset_filter_msg}, parallel={frontier_w}]",
            usable_for="ml_training, ranker training, daily strategies",
            cmd=[PYTHON, "src/pipeline/frontier_consolidator.py",
                 "--workers", str(frontier_w)] + asset_arg +
                (["--universe", universe] if universe and not asset else []) +
                (["--force"] if force else []),
            output_check=lambda: _check_dir_files(_layout.frontier_dir(), "*.parquet"),
            # Stale if any upstream panel (hawkes/panels) is newer
            is_stale=lambda: _is_stale_relative(
                _layout.frontier_dir(),
                _layout.hawkes_dir(),
                _layout.panels_dir(),
            ),
            depends_on=["build_panels", "hawkes_branching"],
        ),
        Stage(
            key="chimera_legacy",
            tier=4,
            description=f"Build v50 chimera ({asset_filter_msg}, parallel={chimera_legacy_w})",
            usable_for="V1-V14 inference + retraining, legacy paper trader",
            cmd=[PYTHON, "src/pipeline/make_dataset_legacy.py",
                 "--workers", str(chimera_legacy_w)] +
                (["--asset", asset] if asset else []) +
                (["--universe", universe] if universe and not asset else []) +
                (["--force"] if force else []),
            output_check=lambda: _check_dir_files(_layout.chimera_legacy_dir(), "*.parquet"),
            is_stale=lambda: _is_stale_relative(_layout.chimera_legacy_dir(), raw_dir,
                                                 upstream_glob="**/aggTrades/*.parquet"),
            depends_on=["fetch_binance"],
        ),
        Stage(
            key="chimera_v51",
            tier=5,
            description=f"Build v51 SOTA chimera + 4 cadence views ({asset_filter_msg}, parallel={chimera_v51_w})",
            usable_for="V0/V1.x f121 training, ChimeraLoader, full strategy stack",
            cmd=[PYTHON, "src/pipeline/make_dataset.py",
                 "--workers", str(chimera_v51_w)] +
                (["--asset", asset] if asset else []) +
                (["--universe", universe] if universe else []) +
                (["--force"] if force else []),
            output_check=lambda: _check_dir_files(_layout.chimera_dir("dollar"), "*.parquet"),
            # Stale if v50 chimera, frontier silver, or hawkes/panels newer
            is_stale=lambda: _is_stale_relative(
                _layout.chimera_dir("dollar"),
                _layout.chimera_legacy_dir(),
                _layout.frontier_dir(),
                _layout.hawkes_dir(),
                _layout.panels_dir(),
            ),
            depends_on=["chimera_legacy", "frontier_consolidate"],
        ),
        Stage(
            key="validate",
            tier=6,
            description="Pre-train gate: 5 validators",
            usable_for="signal that the data is ready for model training",
            cmd=[PYTHON, "src/pipeline/pre_train_gate.py"] +
                (["--asset", asset] if asset else ["--asset", "BTC"]) + ["--quick"],
            output_check=lambda: {"ok": True, "n_files": 1, "latest_mtime": None,
                                   "latest_mtime_epoch": 0.0, "dir": "logs/"},
            # Always re-run validate — it's cheap + checks invariants
            is_stale=lambda: (True, "always re-run"),
            depends_on=["chimera_v51"],
        ),
        Stage(
            key="gc_snapshots",
            tier=7,
            description="GC older dated snapshots (keep newest 1 valid per asset/key)",
            usable_for="freeing disk space; column-aware so corrupt newer files do not displace healthy older fallbacks",
            # Keep newest 1 VALID snapshot — frozen date cutoffs make older
            # snapshots strictly redundant (identical train/val/oos, smaller unseen).
            cmd=[PYTHON, "src/pipeline/gc_snapshots.py", "--keep", "1"],
            output_check=lambda: {"ok": True, "n_files": 1, "latest_mtime": None,
                                   "latest_mtime_epoch": 0.0, "dir": "logs/"},
            # Runs automatically after validate (default-on). Use --no-gc to skip.
            is_stale=lambda: (False, "opt-in via runner gating (default-on after validate)"),
            depends_on=["validate"],
        ),
    ]


# ─── State persistence ────────────────────────────────────────────────────────

def load_state() -> dict:
    if not PIPELINE_STATE_PATH.exists():
        return {"runs": [], "last_status_per_stage": {}}
    try:
        return json.loads(PIPELINE_STATE_PATH.read_text())
    except Exception:
        return {"runs": [], "last_status_per_stage": {}}


def save_state(state: dict) -> None:
    PIPELINE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PIPELINE_STATE_PATH.write_text(json.dumps(state, indent=2))


def update_state(state: dict, stage_key: str, result: dict) -> None:
    state["last_status_per_stage"][stage_key] = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        **result,
    }
    state.setdefault("runs", []).append({
        "stage": stage_key,
        "started_at": result.get("started_at"),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "exit_code": result.get("exit_code"),
        "duration_s": result.get("duration_s"),
    })
    if len(state["runs"]) > 500:
        state["runs"] = state["runs"][-500:]


# ─── Stage execution ──────────────────────────────────────────────────────────

def run_stage(stage: Stage, dry_run: bool = False,
              heartbeat_seconds: int = 30) -> dict:
    """Run one stage with live progress visibility.

    The stage's stdout is tee'd to its dedicated log file. A heartbeat thread
    prints elapsed time + (where applicable) live output-file count every
    `heartbeat_seconds` so you know the stage is making progress.
    """
    import threading

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOG_DIR / f"{stage.key}_{ts}.log"

    # Resolve expected output directory up-front so the user knows where to
    # look for the stage's products before it runs (not just after).
    expected_dir = None
    expected_n = 0
    if stage.output_check:
        try:
            _pre_check = stage.output_check()
            expected_dir = _pre_check.get("dir")
            expected_n = int(_pre_check.get("n_files", 0) or 0)
        except Exception:
            pass

    print(f"\n{'='*72}")
    print(f"[T{stage.tier}] {stage.key}")
    print(f"   {stage.description}")
    print(f"   AFTER COMPLETION usable for: {stage.usable_for}")
    if expected_dir:
        # Print as a project-relative path when possible; absolute otherwise.
        try:
            _rel = Path(expected_dir).resolve().relative_to(PROJECT_ROOT)
            _show = str(_rel)
        except Exception:
            _show = str(expected_dir)
        print(f"   EXPECTED OUTPUT: {_show}  (currently {expected_n} files)")
    print(f"   $ {' '.join(stage.cmd)}")
    if dry_run:
        print(f"   [DRY-RUN] would log to {log_path.relative_to(PROJECT_ROOT)}")
        return {"started_at": datetime.now(timezone.utc).isoformat(),
                "exit_code": 0, "duration_s": 0.0, "log_path": str(log_path), "dry_run": True}
    print(f"   logging to {log_path.relative_to(PROJECT_ROOT)}", flush=True)
    print(f"   heartbeat every {heartbeat_seconds}s (tail -f the log for full output)",
          flush=True)

    started = time.time()
    started_iso = datetime.now(timezone.utc).isoformat()

    # Heartbeat thread: prints periodic progress while the subprocess runs.
    stop_evt = threading.Event()

    def _heartbeat():
        while not stop_evt.wait(heartbeat_seconds):
            elapsed_s = time.time() - started
            # Live file count if the stage has an output_check
            n_files = 0
            try:
                if stage.output_check:
                    out = stage.output_check()
                    n_files = int(out.get("n_files", 0))
            except Exception:
                pass
            # Tail last log line (gives signal of what the inner process is doing)
            tail = ""
            try:
                if log_path.exists() and log_path.stat().st_size > 0:
                    with open(log_path, "rb") as lf:
                        # Read last ~1KB and get final line
                        lf.seek(0, 2)
                        size = lf.tell()
                        lf.seek(max(0, size - 1024))
                        chunk = lf.read().decode("utf-8", errors="replace")
                        lines = [ln for ln in chunk.splitlines() if ln.strip()]
                        if lines:
                            tail = lines[-1].strip()[:90]
            except Exception:
                pass

            # Per-asset progress rollup (mirrors fetch's per-instrument view)
            asset_part = ""
            try:
                prog = parse_per_asset_progress(stage.key, log_path)
                if prog:
                    parts = []
                    parts.append(f"done={prog['n_done']}")
                    if prog["n_running"]:
                        run_show = prog["running"][:3]
                        run_str = ",".join(run_show)
                        if prog["n_running"] > 3:
                            run_str += f"+{prog['n_running'] - 3}"
                        parts.append(f"running=[{run_str}]")
                    if prog["n_err"]:
                        err_show = prog["err"][:3]
                        err_str = ",".join(err_show)
                        parts.append(f"err=[{err_str}]")
                    asset_part = " · " + " ".join(parts)
            except Exception:
                pass

            elapsed_min = elapsed_s / 60.0
            extra = f", n_files={n_files}" if n_files else ""
            tail_part = f" :: {tail}" if tail else ""
            print(f"   [HEARTBEAT] {stage.key} elapsed {elapsed_min:.1f} min"
                  f"{extra}{asset_part}{tail_part}", flush=True)

    hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    hb_thread.start()

    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            r = subprocess.run(stage.cmd, stdout=logf, stderr=subprocess.STDOUT,
                               cwd=str(PROJECT_ROOT), timeout=86400)
        rc = r.returncode
    except subprocess.TimeoutExpired:
        rc = 124
    finally:
        stop_evt.set()
        hb_thread.join(timeout=2.0)

    elapsed = time.time() - started

    result = {
        "started_at": started_iso,
        "exit_code": rc,
        "duration_s": round(elapsed, 1),
        "log_path": str(log_path),
    }

    # Output check
    if stage.output_check:
        try:
            result["output"] = stage.output_check()
        except Exception as e:
            result["output"] = {"ok": False, "err": str(e)}

    marker = "OK" if rc == 0 else f"FAIL exit={rc}"
    print(f"   [{marker}] elapsed {elapsed/60:.1f} min", flush=True)
    if "output" in result:
        o = result["output"]
        if o.get("ok"):
            print(f"   USABLE: {o.get('n_files', 0)} files at {o.get('dir')}", flush=True)
        else:
            print(f"   NO OUTPUTS: {o}", flush=True)
    return result


# ─── CLI ──────────────────────────────────────────────────────────────────────

def cmd_status() -> int:
    state = load_state()
    print("="*80)
    print("PIPELINE STATUS - what's usable right now")
    print("="*80)
    for stage in _stages(asset=None, universe=None, workers=24):
        # Live output check (queries disk, not state file)
        out = {}
        if stage.output_check:
            try:
                out = stage.output_check()
            except Exception as e:
                out = {"ok": False, "err": str(e)}
        status = state["last_status_per_stage"].get(stage.key, {})
        marker = "OK  " if out.get("ok") else "MISS"
        n = out.get("n_files", 0)
        last = status.get("completed_at", "(never via runner)")
        print(f"  [T{stage.tier}] {marker}  {stage.key:<22}  {n:>5} files   last_run={last}")
        print(f"           usable for: {stage.usable_for}")
    print()
    return 0


def parse_tiers(s: str) -> list[str]:
    if s == "all":
        # Reordered 2026-04-28: chimera_legacy moved up to run immediately
        # after fetch_binance. Rationale: chimera_legacy only depends on
        # data/raw aggTrades + funding + metrics — same dependency set as
        # bar_fabric/hawkes/build_panels — so it can run in parallel-ish
        # right after fetch lands. Old order ran the chimera (which V1-V14
        # inference depends on) AFTER 30-90 min of hawkes + frontier silver
        # work; that delay blocked model retraining unnecessarily. New
        # order: V1-V14 chimera ready ~10-20 min post-fetch.
        return ["fetch_binance", "chimera_legacy", "bar_fabric", "build_panels",
                "hawkes_branching", "frontier_consolidate", "chimera_v51",
                "validate"]
    return [t.strip() for t in s.split(",") if t.strip()]


def _delegate_to_refresh(args) -> int:
    """Delegate to src/pipeline/refresh.py with mapped args.

    Maps the legacy --tiers / --asset / --universe / --force surface to
    refresh.py's --target / --asset / --scope / --force. Tier-to-target map:
        all                  -> chimera_v51 (default DAG root)
        validate             -> pre_train_gate
        bar_fabric           -> bar_dib
        build_panels         -> frontier_silver (rebuilds all panels via deps)
        chimera_legacy       -> chimera_legacy
        chimera_v51          -> chimera_v51
        frontier_consolidate -> frontier_silver
        hawkes_branching     -> hawkes_branching_panel
    """
    print("[run_pipeline] --via-refresh: delegating to refresh.py", flush=True)
    selected = parse_tiers(args.tiers)
    # Pick the most-downstream selected stage as the refresh target (DAG runner
    # walks deps automatically).
    tier_to_target = {
        "fetch_binance": "raw_aggtrades",
        "bar_fabric": "bar_dib",
        "build_panels": "frontier_silver",
        "hawkes_branching": "hawkes_branching_panel",
        "frontier_consolidate": "frontier_silver",
        "chimera_legacy": "chimera_legacy",
        "chimera_v51": "chimera_v51",
        "validate": "pre_train_gate",
    }
    # Pick deepest target the user selected (heuristic: validate > chimera_v51 >
    # chimera_legacy > frontier_silver > others).
    priority = ["validate", "chimera_v51", "chimera_legacy", "frontier_silver",
                "build_panels", "hawkes_branching", "bar_fabric", "fetch_binance"]
    target = None
    for t in priority:
        if t in selected:
            target = tier_to_target[t]
            break
    if target is None:
        target = "chimera_v51"

    cmd = [PYTHON, str(PROJECT_ROOT / "src" / "pipeline" / "refresh.py"),
           "--target", target]
    if args.asset:
        # refresh.py exposes --assets (plural, nargs); --asset is rejected.
        cmd.extend(["--assets", args.asset])
    if args.universe:
        cmd.extend(["--scope", args.universe])
    if args.force:
        cmd.append("--force")
    if args.dry_run:
        cmd.append("--dry-run")
    print(f"[run_pipeline] $ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="all",
                    help="Comma-separated stage keys, or 'all' (default: all). "
                         "Available: fetch_binance, bar_fabric, build_panels, "
                         "hawkes_branching, chimera_legacy, frontier_consolidate, "
                         "chimera_v51, validate, gc_snapshots")
    ap.add_argument("--asset", default=None, help="Single asset (e.g. BTC)")
    ap.add_argument("--universe", default=None, choices=["u10", "u50", "u100"],
                    help="Restrict v51 build to a universe")
    ap.add_argument("--workers", type=int, default=24,
                    help="Default parallelism (esp. for fetch). Stage-specific "
                         "overrides via --stage-workers KEY=N (repeatable).")
    ap.add_argument("--stage-workers", action="append", default=[],
                    metavar="KEY=N",
                    help="Per-stage worker override. KEYs: fetch, bars, hawkes, "
                         "frontier, chimera_legacy, chimera_v51. Example: "
                         "--stage-workers fetch=32 --stage-workers frontier=8")
    ap.add_argument("--dry-run", action="store_true", help="Preview without executing")
    ap.add_argument("--status", action="store_true", help="Show usability of each stage")
    ap.add_argument("--continue-on-fail", action="store_true",
                    help="Don't stop on a failed stage (default: stop)")
    ap.add_argument("--force", action="store_true",
                    help="Re-run stages even when their is_stale check says fresh "
                         "(default: skip up-to-date stages)")
    ap.add_argument("--heartbeat-seconds", type=int, default=30,
                    help="Print a heartbeat (elapsed + file count + log tail) every "
                         "N seconds while a stage runs (default: 30).")
    ap.add_argument("--no-gc", action="store_true",
                    help="Disable the default post-validate GC. By default, after "
                         "validate passes the runner deletes older dated snapshots "
                         "(keep newest 1 valid per asset/key). Safe because GC is "
                         "column-aware: corrupt newer files do not displace healthy "
                         "older fallbacks. With frozen split dates, train/val/oos "
                         "are identical run-to-run; only unseen grows -- older "
                         "snapshots add no information.")
    ap.add_argument("--via-refresh", action="store_true",
                    help="Delegate to the new DAG runner (src/pipeline/refresh.py). "
                         "Smarter incremental rebuild: hashes deps and skips fresh "
                         "stages. Recommended path for new work; old --tiers path "
                         "kept for compat during transition.")
    args = ap.parse_args()

    # New-architecture delegation: opt-in via --via-refresh.
    if args.via_refresh:
        return _delegate_to_refresh(args)

    if args.status:
        return cmd_status()

    selected = parse_tiers(args.tiers)
    # GC runs by default after validate (idempotent if user explicitly added it).
    # --no-gc opts out. Skip GC if validate isn't in selected (no fresh data to clean).
    gc_enabled = (not args.no_gc) and ("validate" in selected) and ("gc_snapshots" not in selected)
    if gc_enabled:
        selected.append("gc_snapshots")
    state = load_state()
    # Parse --stage-workers KEY=N overrides
    stage_workers = {}
    for spec in args.stage_workers:
        if "=" not in spec:
            print(f"[error] bad --stage-workers '{spec}', want KEY=N", file=sys.stderr)
            return 2
        k, v = spec.split("=", 1)
        try:
            stage_workers[k.strip()] = int(v.strip())
        except ValueError:
            print(f"[error] bad --stage-workers value '{v}'", file=sys.stderr)
            return 2
    all_stages = _stages(asset=args.asset, universe=args.universe,
                          workers=args.workers, stage_workers=stage_workers,
                          force=args.force)
    stage_map = {s.key: s for s in all_stages}

    # Validate selection
    unknown = [t for t in selected if t not in stage_map]
    if unknown:
        print(f"[error] unknown tiers: {unknown}", file=sys.stderr)
        return 2

    print(f"\n{'='*72}")
    print(f"PIPELINE RUNNER - selected: {selected}")
    print(f"  asset={args.asset}, universe={args.universe}, workers={args.workers}")
    print(f"  dry_run={args.dry_run}, continue_on_fail={args.continue_on_fail}")
    print(f"{'='*72}")

    overall_started = time.time()
    n_ok = n_fail = n_skipped = 0
    for key in selected:
        stage = stage_map[key]

        # Dependency check (informational only; we don't enforce)
        for dep in stage.depends_on:
            dep_status = state.get("last_status_per_stage", {}).get(dep, {})
            if not dep_status.get("output", {}).get("ok"):
                # Also check disk directly via the dep's output_check
                dep_stage = stage_map.get(dep)
                if dep_stage and dep_stage.output_check:
                    try:
                        if dep_stage.output_check().get("ok"):
                            continue  # dep ran outside the runner; OK
                    except Exception:
                        pass
                print(f"[warn] {key}: depends on {dep} which appears not yet run; "
                      f"continuing anyway")

        # Staleness check — skip if fresh and not --force.
        # gc_snapshots is opt-in via is_stale=False; force it whenever it lands
        # in `selected` (either auto-added post-validate, or explicit --tiers).
        force_this = args.force or (stage.key == "gc_snapshots")
        if not force_this and stage.is_stale and not args.dry_run:
            try:
                stale, reason = stage.is_stale()
            except Exception as e:
                stale, reason = True, f"is_stale error: {e}"
            if not stale:
                print(f"\n{'='*72}")
                print(f"[T{stage.tier}] {stage.key}")
                print(f"   [SKIP-FRESH] {reason}")
                print(f"   (use --force to re-run anyway)")
                n_skipped += 1
                continue

        result = run_stage(stage, dry_run=args.dry_run,
                           heartbeat_seconds=args.heartbeat_seconds)
        update_state(state, key, result)
        save_state(state)
        if result["exit_code"] != 0:
            n_fail += 1
            if not args.continue_on_fail:
                print(f"\n[STOP] {key} failed (exit {result['exit_code']}). "
                      f"Use --continue-on-fail to keep going.")
                break
        else:
            n_ok += 1

    overall_elapsed = time.time() - overall_started
    print(f"\n{'='*72}")
    print(f"PIPELINE RUNNER FINISHED - {n_ok} ok, {n_fail} fail, "
          f"{n_skipped} skipped (fresh), elapsed {overall_elapsed/60:.1f} min")
    print(f"  state: {PIPELINE_STATE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"{'='*72}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
