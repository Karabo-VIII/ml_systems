"""Pipeline DAG runner — thin orchestrator over self-sufficient producers.

REFACTORED 2026-05-21. Producers now own their own:
  * Asset filtering        (--assets plural; canonical via src/pipeline/cli.py)
  * Universe resolution    (--universe u10/u50/u100)
  * Skip-existing          (mtime/delta-state/done-keys per stage)
  * Worker parallelism     (--workers; per-stage --max-workers cap in DAG yaml)
  * Force rebuild          (--force)

refresh.py's job is the *narrow* orchestration:
  1. Load `config/asset_dag.yaml` -> topo-walk deps
  2. For each stage in order: call producer with canonical flags
  3. Validate output against the gate string
  4. Catch STUB / PARTIAL_STUB silent-failure cases
  5. Serialize memory_exclusive stages (chimera_legacy, chimera_v51)
  6. Log failures + emit live heartbeat for monitoring
  7. Provide --status / --live / --list / --failures viewers

Removed in the refactor:
  * `compute_content_hash` cache (~80 lines) — producers do their own skip
  * `_check_freshness_skip` + --respect-freshness (~35 lines) — same
  * Per-asset iteration loop (~150 lines) — `--assets` plural handles batching
  * `_PASSTHROUGH_ARGS` (~80 lines) — canonical CLI eliminates need
  * --smart-force (was hash-gated) — superseded by producer skip-existing
  * --start-from-wave / --only-waves / --skip-waves — niche; use --exclude

Result: ~1820 lines -> ~720 lines (60% reduction).

CLI (canonical post-refactor):
  python src/pipeline/refresh.py --all
  python src/pipeline/refresh.py --target chimera_v51 --universe u50
  python src/pipeline/refresh.py --target hawkes_branching_panel --assets BTC ETH
  python src/pipeline/refresh.py --target chimera_v51 --force --parallel
  python src/pipeline/refresh.py --status --target chimera_v51
  python src/pipeline/refresh.py --live           # NRT monitor of running job
  python src/pipeline/refresh.py --failures       # tail _failures.log
  python src/pipeline/refresh.py --list           # show DAG
"""
from __future__ import annotations

__contract__ = {
    "kind": "pipeline_orchestrator",
    "stage": "refresh",
    "inputs": {
        "args": ["--target", "--all", "--assets", "--universe", "--force",
                 "--workers", "--parallel", "--exclude", "--no-deps",
                 "--dry-run", "--status", "--list", "--live", "--failures"],
        "config": "config/asset_dag.yaml",
    },
    "outputs": {
        "state_file": "data/_dag_state.json",
        "live_file": "data/_dag_state_live.json",
        "logs": "logs/refresh/<stage>_<timestamp>.log",
        "failures": "logs/refresh/_failures.log",
    },
    "invariants": {
        "explicit_dag_no_implicit_ordering": True,
        "producer_handles_skip_existing": True,
        "per_stage_gate": True,
        "memory_exclusive_serialized": True,
    },
}

import argparse
import json
import os
import subprocess
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

# Tracks the currently-running producer subprocess so the signal handler can
# kill the entire process tree (Popen child + its grandchildren). Without
# this, Ctrl-C / taskkill leaves orphans holding multi-GB RAM (audited
# 2026-05-23: PID 27432 alive 4h after a killed refresh, 2.96 GB resident).
_RUNNING_PROC = None  # type: Optional[object]  # subprocess.Popen
_SIGNAL_HANDLERS_INSTALLED = False


def _kill_proc_tree(pid: int, timeout: float = 5.0) -> None:
    """Cross-platform process-tree kill via psutil.

    Windows subprocess.Popen.kill() only kills the immediate child; multiprocessing
    grandchildren (fetch_all workers, chimera ProcessPool) survive. psutil traverses
    the tree.
    """
    try:
        import psutil
    except ImportError:
        return
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    procs = []
    try:
        procs = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        pass
    procs.append(parent)
    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    gone, alive = psutil.wait_procs(procs, timeout=timeout)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass


def _cleanup_on_exit() -> None:
    """Best-effort cleanup: kill tracked subprocess tree, clear live state.

    The lock file is owned by `_RefreshLock.__exit__` and cleans itself.
    """
    global _RUNNING_PROC
    proc = _RUNNING_PROC
    _RUNNING_PROC = None
    if proc is not None:
        try:
            if proc.poll() is None:
                _kill_proc_tree(proc.pid)
        except Exception:
            pass
    try:
        if LIVE_PATH.exists():
            LIVE_PATH.unlink()
    except OSError:
        pass


def _install_signal_handlers() -> None:
    """Register SIGINT / SIGTERM / SIGBREAK -> tree-kill + state cleanup -> exit.

    Idempotent. Without this, Ctrl-C leaks subprocesses and freezes
    _dag_state_live.json mid-stage.
    """
    global _SIGNAL_HANDLERS_INSTALLED
    if _SIGNAL_HANDLERS_INSTALLED:
        return

    def _handler(signum, frame):
        try:
            sys.stderr.write(f"\n[refresh] caught signal {signum}; "
                             f"killing subprocess tree + clearing live state...\n")
            sys.stderr.flush()
        except Exception:
            pass
        _cleanup_on_exit()
        # 130 for SIGINT, 143 for SIGTERM (POSIX conventions; meaningful enough on Windows)
        sys.exit(130 if signum == signal.SIGINT else 143)

    try:
        signal.signal(signal.SIGINT, _handler)
    except (ValueError, OSError):
        pass
    try:
        signal.signal(signal.SIGTERM, _handler)
    except (ValueError, OSError, AttributeError):
        pass
    if os.name == "nt":
        try:
            # SIGBREAK fires on Ctrl-Break (Windows-specific)
            signal.signal(signal.SIGBREAK, _handler)
        except (ValueError, OSError, AttributeError):
            pass
    _SIGNAL_HANDLERS_INSTALLED = True

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DAG_PATH = PROJECT_ROOT / "config" / "asset_dag.yaml"
STATE_PATH = PROJECT_ROOT / "data" / "_dag_state.json"
LIVE_PATH = PROJECT_ROOT / "data" / "_dag_state_live.json"
LOCK_PATH = PROJECT_ROOT / "data" / "_dag_state.lock"
FAILURES_LOG = PROJECT_ROOT / "logs" / "refresh" / "_failures.log"
LOG_DIR = PROJECT_ROOT / "logs" / "refresh"
PYTHON = sys.executable


# ─── DAG model ──────────────────────────────────────────────────────────────

@dataclass
class AssetSpec:
    name: str
    producer: str
    producer_args: list = field(default_factory=list)
    output: str = ""
    output_kind: str = "single_file"
    deps: list = field(default_factory=list)
    per_asset: bool = False
    universe_aware: bool = False
    freshness: str = "daily"
    gate: Optional[str] = None
    notes: str = ""
    accepts_rc1: bool = False
    max_workers: int = 0
    timeout_seconds: int = 14400
    memory_exclusive: bool = False


def load_dag(path: Path = DAG_PATH) -> dict[str, AssetSpec]:
    """Parse config/asset_dag.yaml into {name: AssetSpec}."""
    with open(path) as f:
        spec = yaml.safe_load(f)
    out: dict[str, AssetSpec] = {}
    for name, body in (spec.get("assets") or {}).items():
        out[name] = AssetSpec(
            name=name,
            producer=body.get("producer", ""),
            producer_args=list(body.get("producer_args", [])),
            output=body.get("output", ""),
            output_kind=body.get("output_kind", "single_file"),
            deps=list(body.get("deps", [])),
            per_asset=bool(body.get("per_asset", False)),
            universe_aware=bool(body.get("universe_aware", False)),
            freshness=body.get("freshness", "daily"),
            gate=body.get("gate"),
            notes=body.get("notes", ""),
            accepts_rc1=bool(body.get("accepts_rc1", False)),
            max_workers=int(body.get("max_workers", 0)),
            timeout_seconds=int(body.get("timeout_seconds", 14400)),
            memory_exclusive=bool(body.get("memory_exclusive", False)),
        )
    return out


def topo_walk(dag: dict[str, AssetSpec], target: str) -> list[str]:
    """Return ordered list of stages to build (target last). Skips out-of-DAG deps."""
    if target not in dag:
        raise ValueError(f"unknown target: {target!r}; known: {sorted(dag.keys())}")
    visited: set[str] = set()
    in_progress: set[str] = set()
    order: list[str] = []
    def dfs(name: str):
        if name in visited:
            return
        if name in in_progress:
            # Back-edge => cycle. Fail loudly instead of silently producing a
            # partial / wrong build order.
            raise ValueError(f"cycle detected in asset DAG at stage {name!r}")
        in_progress.add(name)
        for d in dag[name].deps:
            if d in dag:
                dfs(d)
        in_progress.discard(name)
        visited.add(name)
        order.append(name)
    dfs(target)
    return order


def topo_walk_all(dag: dict[str, AssetSpec]) -> list[str]:
    """Topological order over EVERY stage in the DAG."""
    visited: set[str] = set()
    order: list[str] = []
    def dfs(name: str):
        if name in visited: return
        visited.add(name)
        for d in dag[name].deps:
            if d in dag:
                dfs(d)
        order.append(name)
    for name in sorted(dag.keys()):
        dfs(name)
    return order


def topo_waves(dag: dict[str, AssetSpec], walk: list[str]) -> list[list[str]]:
    """Group walk into independent-dep waves for parallel execution.

    Within a wave, stages can run concurrently. memory_exclusive stages are
    extracted into their own singleton waves to avoid OOM.
    """
    walk_set = set(walk)
    pending = list(walk)
    done: set[str] = set()
    waves: list[list[str]] = []
    while pending:
        wave: list[str] = []
        for stage_name in list(pending):
            spec = dag[stage_name]
            deps_in_walk = [d for d in spec.deps if d in dag and d in walk_set]
            if all(d in done for d in deps_in_walk):
                wave.append(stage_name)
        if not wave:
            waves.append(list(pending))
            break
        # Split memory_exclusive stages into singleton waves
        excl = [s for s in wave if dag[s].memory_exclusive]
        normal = [s for s in wave if not dag[s].memory_exclusive]
        if normal:
            waves.append(normal)
            for s in normal: pending.remove(s); done.add(s)
        for s in excl:
            waves.append([s])
            pending.remove(s); done.add(s)
    return waves


# ─── State + live file ──────────────────────────────────────────────────────

def load_state() -> dict:
    if not STATE_PATH.exists():
        return {"stages": {}}
    try:
        d = json.loads(STATE_PATH.read_text())
        # Migrate legacy "assets" key to "stages"
        if "stages" not in d and "assets" in d:
            d["stages"] = d.pop("assets")
        return d
    except Exception:
        return {"stages": {}}


# Serializes concurrent state/heartbeat writes within this process. Under --parallel,
# multiple stage threads call write_live() at once; the old (unlink + tmp.rename)
# pattern raced on the SHARED tmp path and used os.rename (which on Windows CANNOT
# overwrite a locked/existing target) -> WinError 32 crashed a 5h refresh (2026-05-31).
_STATE_LOCK = threading.Lock()


def _atomic_json_write(path, payload: dict, *, best_effort: bool = False,
                       attempts: int = 5) -> None:
    """Thread-safe atomic JSON write via os.replace (atomic overwrite on Windows).

    Per-thread tmp name (no shared-tmp race) + a process lock + a short retry for the
    transient case where a reader (e.g. `refresh.py --live`) holds the target. When
    best_effort=True a persistent lock is SWALLOWED (the heartbeat is monitoring-only;
    a failed beat must never crash the run).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, default=str)
    with _STATE_LOCK:
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.json.tmp")
        last = None
        for i in range(attempts):
            try:
                tmp.write_text(data)
                os.replace(tmp, path)
                return
            except (PermissionError, OSError) as e:
                last = e
                time.sleep(0.02 * (i + 1))
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        if not best_effort and last is not None:
            raise last


def save_state(state: dict) -> None:
    _atomic_json_write(STATE_PATH, state, best_effort=True)


def write_live(payload: dict) -> None:
    """Best-effort atomic write of the always-current heartbeat file (never fatal)."""
    _atomic_json_write(LIVE_PATH, payload, best_effort=True)


def append_failure(stage: str, status: str, msg: str, log_path: str = "") -> None:
    """Append a failure record to logs/refresh/_failures.log."""
    FAILURES_LOG.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    with open(FAILURES_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] stage={stage} status={status} msg={msg!r} log={log_path}\n")


def _tail_log_lines(log_path: Path, n: int = 5) -> list[str]:
    if not log_path.exists():
        return []
    try:
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 16384))
            data = f.read().decode("utf-8", errors="replace")
        data = data.replace("\r", "\n")
        lines = [ln.strip() for ln in data.splitlines() if ln.strip()]
        return lines[-n:]
    except Exception:
        return []


def _file_mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def detect_active_run(stale_threshold_sec: float = 120.0) -> Optional[dict]:
    """Return the live state if another refresh process is currently running."""
    if not LIVE_PATH.exists():
        return None
    try:
        state = json.loads(LIVE_PATH.read_text())
    except Exception:
        return None
    if "summary" in state:
        return None
    last_update = state.get("now_utc")
    if not last_update:
        return None
    try:
        last_dt = datetime.fromisoformat(last_update.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    age_s = (datetime.now(timezone.utc) - last_dt).total_seconds()
    if age_s > stale_threshold_sec:
        return None
    return state


class _RefreshLock:
    """OS-level exclusive lock on data/_dag_state.lock.

    Closes the check-then-act race window in `detect_active_run` (CRIT-1 from
    Opus DAG audit ab3a3e41777c11fde, 2026-05-22). Without this, two
    `refresh.py` invocations within ~2s both pass the LIVE_PATH check and
    proceed to run concurrent producers — silent data corruption possible
    on atomic-tmp-rename collision.

    Acquire BEFORE writing live state; release on context exit. Cross-platform
    (msvcrt on Windows, fcntl on POSIX). Non-blocking by default.
    """
    def __init__(self, path: Path = LOCK_PATH):
        self.path = path
        self.fp = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = open(self.path, "a+")
        try:
            if os.name == "nt":  # Windows
                import msvcrt
                # locking() raises OSError if held by another process
                msvcrt.locking(self.fp.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # POSIX
                import fcntl
                fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, ImportError) as e:
            self.fp.close()
            self.fp = None
            raise RuntimeError(
                f"refresh.py concurrent-run lock failed at {self.path}: {e}. "
                f"Another `refresh.py` is running. Use `--status` to inspect, "
                f"or wait for it to complete."
            ) from e
        # Write owner pid + start time for forensics
        try:
            self.fp.seek(0)
            self.fp.truncate()
            self.fp.write(f"pid={os.getpid()} started_utc={datetime.now(timezone.utc).isoformat()}\n")
            self.fp.flush()
        except Exception:
            pass
        return self

    def __exit__(self, *args):
        if self.fp is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                self.fp.seek(0)
                try:
                    msvcrt.locking(self.fp.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                import fcntl
                fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
        finally:
            self.fp.close()
            try:
                self.path.unlink()
            except OSError:
                pass


# ─── Output materialization (for STUB detection + gate) ─────────────────────

def output_paths(spec: AssetSpec) -> list[Path]:
    """Resolve canonical output(s) for a stage. {asset} -> wildcard."""
    pattern = spec.output.replace("{asset}", "*")
    p = PROJECT_ROOT / pattern
    pat_str = str(p)
    if any(c in pat_str for c in "*?["):
        parts = list(p.parts)
        glob_start = None
        for i, part in enumerate(parts):
            if any(c in part for c in "*?["):
                glob_start = i
                break
        if glob_start is None:
            return []
        base = Path(*parts[:glob_start])
        rel_pattern = "/".join(parts[glob_start:])
        try:
            return sorted(base.glob(rel_pattern))
        except Exception:
            return []
    return [p] if p.exists() else []


def is_materialized(spec: AssetSpec) -> bool:
    if spec.output_kind == "ephemeral":
        return True
    return len(output_paths(spec)) > 0


# ─── Producer invocation ────────────────────────────────────────────────────

def build_cmd(spec: AssetSpec, *, universe: Optional[str],
               assets: Optional[list[str]], force: bool,
               workers_override: Optional[int]) -> list[str]:
    """Build the subprocess command for a producer using the canonical CLI.

    Every producer now accepts (via src/pipeline/cli.py or its own argparse):
      --workers N      --force      --universe u10/u50/u100      --assets SYM...
    Per-asset producers route --assets internally; cross-section producers
    accept it as a no-op for contract uniformity.
    """
    cmd = [PYTHON, str(PROJECT_ROOT / spec.producer)] + list(spec.producer_args)

    # --workers override: only patches stages whose producer_args already
    # declare --workers (signaling support); honors per-stage max_workers cap.
    if workers_override is not None and "--workers" in cmd:
        idx = cmd.index("--workers")
        if idx + 1 < len(cmd):
            effective = workers_override
            if spec.max_workers > 0 and effective > spec.max_workers:
                print(f"  [workers-cap] {spec.name}: {effective} -> {spec.max_workers}",
                      flush=True)
                effective = spec.max_workers
            cmd[idx + 1] = str(effective)

    # --universe: only for universe_aware stages
    if spec.universe_aware and universe and "--universe" not in cmd:
        cmd.extend(["--universe", universe])

    # --assets: pass through to per_asset stages OR if the producer documents
    # acceptance (cross-section producers accept it as no-op per the contract).
    # Skip if cross-section ephemeral (e.g. pre_train_gate) — no asset semantics.
    if assets and spec.per_asset and spec.output_kind != "ephemeral":
        cmd.extend(["--assets"] + list(assets))

    if force:
        cmd.append("--force")
    return cmd


def invoke_producer(spec: AssetSpec, cmd: list[str], log_path: Path,
                     heartbeat_cb=None, heartbeat_seconds: int = 30,
                     stream_to_terminal: bool = True) -> tuple[int, float, set, set]:
    """Run the producer subprocess with live heartbeat AND terminal streaming.

    Returns (rc, elapsed_s, all_present_outputs, fresh_outputs).

    Changed 2026-05-22 (oracle pipeline-progress closure): previously the
    subprocess piped stdout DIRECTLY to log_path via Popen(stdout=logf),
    which meant the user's terminal saw NOTHING from the producer — only
    the heartbeat file (data/_dag_state_live.json) got updates.

    Now: stdout is captured via PIPE and a single reader thread tees each
    line to BOTH the log file AND sys.stdout. The user sees the producer's
    rich progress in real time (basis_signals' [basis_feat scan i/N],
    make_dataset's tqdm bars, fetch_all's [DL]/[OK]/[SKIP] markers) while
    the log file and heartbeat behavior are preserved.

    stream_to_terminal=False reverts to log-only behavior for CI / quiet
    runs (refresh.py --quiet would set this).
    """
    import threading
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pre_run_ts = t0

    # Tag prefix shown in terminal so the user can identify the producer
    # output amid refresh.py's own emissions. The log file gets raw output
    # (no prefix) so log-analysis tools see what the producer printed.
    term_prefix = f"  [{spec.name}] " if stream_to_terminal else ""

    rc = -1
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(f"# refresh.py invocation\n# cmd: {' '.join(cmd)}\n"
                    f"# t0: {datetime.now(timezone.utc).isoformat()}\n\n")
        logf.flush()

        proc = None
        reader_thread = None
        global _RUNNING_PROC
        try:
            popen_kwargs = dict(
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                bufsize=1,           # line-buffered
                universal_newlines=True,
                encoding="utf-8",
                errors="replace",
            )
            # On Windows, give the producer its own process group so we can
            # send CTRL_BREAK to the whole tree if needed. Also lets psutil
            # walk children reliably (Job Object would be stronger but psutil
            # works for our subprocess shapes — Popen child + multiprocessing
            # grandchildren).
            if os.name == "nt":
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            proc = subprocess.Popen(cmd, **popen_kwargs)
            _RUNNING_PROC = proc

            def _reader() -> None:
                """Tee subprocess stdout: log file + (optional) terminal."""
                if proc is None or proc.stdout is None:
                    return
                for line in proc.stdout:
                    # Always write raw to log file
                    try:
                        logf.write(line)
                        logf.flush()
                    except Exception:
                        pass
                    # Optionally echo to terminal with module prefix
                    if stream_to_terminal:
                        try:
                            # Strip trailing newline so prefix shows clean,
                            # then add it back. tqdm uses \r — pass through.
                            stripped = line.rstrip("\n")
                            print(f"{term_prefix}{stripped}", flush=True)
                        except Exception:
                            pass

            reader_thread = threading.Thread(target=_reader, daemon=True)
            reader_thread.start()

            last_hb = t0
            while True:
                rc = proc.poll()
                if rc is not None:
                    break
                now = time.time()
                if heartbeat_cb is not None and (now - last_hb) >= heartbeat_seconds:
                    try:
                        tail = _tail_log_lines(log_path, n=5)
                        heartbeat_cb(elapsed_s=now - t0, log_tail=tail)
                    except Exception:
                        pass
                    last_hb = now
                if (now - t0) > spec.timeout_seconds:
                    proc.kill()
                    rc = 124
                    break
                time.sleep(1.0)
        except Exception as e:
            logf.write(f"\n# refresh.py wrapper exception: {type(e).__name__}: {e}\n")
            rc = -1
        finally:
            # Kill the subprocess tree if still running (KeyboardInterrupt,
            # timeout, exception, or signal handler short-circuit). Without
            # this, the producer + its multiprocessing grandchildren survive
            # the orchestrator's exit.
            if proc is not None and proc.poll() is None:
                try:
                    _kill_proc_tree(proc.pid)
                    rc = proc.poll() if proc.poll() is not None else -1
                except Exception:
                    pass
            _RUNNING_PROC = None
            # Drain remaining output before returning
            if reader_thread is not None:
                reader_thread.join(timeout=5.0)

    all_present = set(output_paths(spec))
    fresh: set = set()
    for p in all_present:
        try:
            if p.stat().st_mtime >= pre_run_ts - 1.0:
                fresh.add(p)
        except OSError:
            continue
    return rc, time.time() - t0, all_present, fresh


# ─── Gate validation ────────────────────────────────────────────────────────

KNOWN_PREFIXES = (
    "xd_", "te_", "rv_", "hbr_", "bs_", "liq_", "mv_", "lob_", "target_",
    "norm_", "wh_", "stbl_", "etf_", "soc_", "fp_", "mvf_", "dv_", "is_",
    "xrel_", "xex_", "bd_", "s3_", "tick_", "bar_",
)


def run_gate(spec: AssetSpec, log_path: Optional[Path] = None) -> tuple[bool, str]:
    """Per-stage gate: read materialized output, verify required cols present.

    Ephemeral stages: per CRIT-2 fix (Opus DAG audit ab3a3e41777c11fde,
    2026-05-22), require a `[gate] FINAL VERDICT` sentinel line in the log
    to prove the producer actually ran to completion. Without it, a
    silent rc=0 crash would falsely report BUILT.
    """
    if spec.output_kind == "ephemeral":
        if log_path is None or not log_path.exists():
            return True, "ok (ephemeral; rc-based gate; no log to sentinel-check)"
        try:
            # Read last ~16KB of log (sentinel should be at the tail)
            with open(log_path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 16384))
                tail = f.read().decode("utf-8", errors="ignore")
        except Exception as e:
            # A log-read failure must NOT silently pass the gate -- we cannot
            # confirm the completion sentinel, so treat it as not-verified.
            return False, (f"ephemeral gate sentinel-check FAILED to read log "
                           f"({type(e).__name__}: {e}); treating as not-verified")
        # Accept any of these patterns (producers may use different wording)
        sentinels = ("[gate] FINAL VERDICT", "GATE_VERDICT:", "[pre_train_gate] DONE",
                     "VALIDATION_COMPLETE", "FINAL VERDICT")
        if not any(s in tail for s in sentinels):
            return False, ("ephemeral stage rc=0 but no completion sentinel in log "
                           f"(searched: {sentinels[:3]}...). Possible silent no-op.")
        return True, "ok (ephemeral; sentinel verified)"
    paths = output_paths(spec)
    if not paths:
        return False, "no output materialized"
    target = sorted(paths, key=lambda p: p.stat().st_mtime)[-1]
    try:
        import polars as pl
        cols = set(pl.read_parquet_schema(target).keys())
    except Exception as e:
        return False, f"schema read failed: {type(e).__name__}: {e}"
    if not spec.gate:
        return True, f"ok ({len(cols)} cols, no explicit gate)"

    import re
    needed_exact: list[str] = []
    needed_prefix: list[str] = []
    for tok in spec.gate.replace(",", " ").replace(";", " ").split():
        clean = tok.rstrip(".,;:")
        if clean.endswith("*"):
            prefix = clean.rstrip("*")
            if prefix.startswith(KNOWN_PREFIXES):
                needed_prefix.append(prefix)
        elif clean.startswith(KNOWN_PREFIXES):
            is_bare_prefix = any(
                clean == p or (clean == p.rstrip("_") + "_") for p in KNOWN_PREFIXES
            )
            if not is_bare_prefix:
                needed_exact.append(clean)
    if "prefix" in spec.gate.lower():
        for tok in spec.gate.replace(",", " ").replace(";", " ").split():
            clean = tok.rstrip(".,;:")
            if clean.endswith("_") and len(clean) >= 3 and clean[:-1].replace("_", "").isalnum():
                needed_prefix.append(clean)

    missing_exact = [c for c in needed_exact if c not in cols]
    missing_prefix = [p for p in needed_prefix if not any(c.startswith(p) for c in cols)]
    if missing_exact or missing_prefix:
        msgs = []
        if missing_exact:
            msgs.append(f"missing cols: {missing_exact[:5]}")
        if missing_prefix:
            msgs.append(f"missing prefixes: {missing_prefix[:5]}")
        return False, "; ".join(msgs)
    return True, f"ok ({len(cols)} cols, all required present)"


# ─── Status viewer ──────────────────────────────────────────────────────────

def status(target: Optional[str] = None) -> None:
    """Print per-stage freshness table for the DAG (or just target's transitive deps)."""
    dag = load_dag()
    state = load_state()
    walk = topo_walk_all(dag) if not target else topo_walk(dag, target)
    print(f"\n{'STAGE':<28} {'STATUS':<10} {'AGE':<12} {'OUTPUT':<60}")
    print("-" * 110)
    for name in walk:
        spec = dag[name]
        rec = state.get("stages", {}).get(name, {})
        last_run_iso = rec.get("last_run", "")
        last_status = rec.get("status", "—")
        if last_run_iso:
            try:
                lr = datetime.fromisoformat(last_run_iso.replace("Z", "+00:00"))
                age_s = (datetime.now(timezone.utc) - lr).total_seconds()
                if age_s < 3600:
                    age = f"{age_s/60:.0f}m"
                elif age_s < 86400:
                    age = f"{age_s/3600:.1f}h"
                else:
                    age = f"{age_s/86400:.1f}d"
            except Exception:
                age = "?"
        else:
            age = "never"
        mat = is_materialized(spec)
        mat_str = "OK" if mat else "MISSING"
        out_short = (spec.output[:55] + "...") if len(spec.output) > 58 else spec.output
        print(f"{name:<28} {last_status:<10} {age:<12} [{mat_str}] {out_short}")
    print()


def _live_loop(*, once: bool, interval: float, stale_s: int,
                clear_screen: bool = True) -> int:
    """Continuously poll _dag_state_live.json and render a snapshot."""
    while True:
        if not LIVE_PATH.exists():
            print(f"[live] no live state at {LIVE_PATH.relative_to(PROJECT_ROOT)}; "
                  f"is a refresh running?")
            return 0
        try:
            state = json.loads(LIVE_PATH.read_text())
        except Exception as e:
            print(f"[live] read failed: {e}")
            return 2

        now = datetime.now(timezone.utc)
        last_iso = state.get("now_utc", "")
        try:
            last_dt = datetime.fromisoformat(last_iso.replace("Z", "+00:00"))
            age_s = (now - last_dt).total_seconds()
        except Exception:
            age_s = 99999
        is_stale = age_s > stale_s

        if clear_screen and not once:
            print("\033[2J\033[H", end="")

        print(f"\n=== refresh.py LIVE @ {now.isoformat(timespec='seconds')} "
              f"(state age {age_s:.0f}s{', STALE' if is_stale else ''}) ===")
        print(f"  target={state.get('target')} universe={state.get('universe')} "
              f"assets={state.get('assets')}")
        print(f"  elapsed={state.get('elapsed_s', 0):.0f}s  "
              f"stages: {state.get('stages_done',0)}/{state.get('stages_total',0)}")
        cur = state.get("current_stage") or {}
        if cur:
            print(f"  current: {cur.get('name')} [{cur.get('status')}] "
                  f"({cur.get('elapsed_s',0):.0f}s)")
            for ln in (cur.get("log_tail") or [])[-3:]:
                print(f"      | {ln[:150]}")
        completed = state.get("completed") or []
        if completed:
            print(f"  completed ({len(completed)}):")
            for c in completed[-5:]:
                print(f"    {c.get('name'):<28} {c.get('status'):<10} "
                      f"{c.get('elapsed_s',0):.0f}s")
        summary = state.get("summary")
        if summary:
            print(f"\n=== RUN COMPLETED ===")
            print(f"  {summary}")
            return 0
        if once:
            return 0
        if is_stale:
            print(f"[live] state file stale (>{stale_s}s); orchestrator likely dead.")
            return 2
        time.sleep(interval)


# ─── Main refresh orchestration ─────────────────────────────────────────────

def refresh(target: Optional[str], *,
             assets: Optional[list[str]] = None,
             universe: Optional[str] = None,
             force: bool = False,
             dry_run: bool = False,
             workers_override: Optional[int] = None,
             refresh_all: bool = False,
             exclude: Optional[list[str]] = None,
             no_deps: bool = False,
             parallel: bool = False,
             max_concurrent_stages: int = 4) -> dict:
    """Walk the DAG, invoke each producer once with canonical flags.

    Producers handle:
      * Asset filtering via --assets
      * Skip-existing internally (no orchestrator-level cache)
      * Parallel workers internally
    refresh.py adds: DAG order, gate validation, STUB detection,
    memory_exclusive serialization, failure logging, live heartbeat.
    """
    dag = load_dag()
    state = load_state()

    if refresh_all:
        walk = topo_walk_all(dag)
        target_label = "ALL"
    else:
        if target is None:
            raise ValueError("refresh(): either target or refresh_all must be set")
        walk = topo_walk(dag, target)
        target_label = target

    if no_deps:
        if refresh_all:
            raise ValueError("--no-deps is incompatible with --all")
        deps_in_walk = [s for s in walk if s != target]
        missing = [s for s in deps_in_walk if not is_materialized(dag[s])]
        if missing:
            print(f"[refresh] HALT: --no-deps requires deps materialized; missing: {missing}",
                  flush=True)
            return {"target": target_label,
                    "results": [{"stage": target, "status": "GATE_FAILED",
                                  "detail": f"missing deps on disk: {missing}"}]}
        walk = [target]
        print(f"[refresh] --no-deps: skipping {len(deps_in_walk)} dep stage(s); "
              f"running only [{target}]", flush=True)

    if exclude:
        unknown = [s for s in exclude if s not in dag]
        if unknown:
            raise ValueError(f"--exclude: unknown stage(s) {unknown}; "
                              f"known: {sorted(dag.keys())}")
        n_before = len(walk)
        walk = [s for s in walk if s not in set(exclude)]
        print(f"[refresh] --exclude dropped {n_before - len(walk)} stage(s): "
              f"{sorted(set(exclude))}")
        for stage_name in exclude:
            if not is_materialized(dag[stage_name]):
                print(f"[refresh] WARN: excluded '{stage_name}' has no materialized "
                      f"output; downstream may use stale data", flush=True)

    print(f"\n{'='*72}")
    workers_str = f"  workers={workers_override}" if workers_override is not None else ""
    par_str = f"  parallel=true(max={max_concurrent_stages})" if parallel else ""
    assets_str = f"  assets={assets}" if assets else ""
    print(f"REFRESH  target={target_label}  universe={universe or '*'}{assets_str}  "
          f"force={force}  dry_run={dry_run}{workers_str}{par_str}")
    print(f"DAG walk ({len(walk)} stages): {' -> '.join(walk)}")
    print(f"  live status: {LIVE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  failures:    {FAILURES_LOG.relative_to(PROJECT_ROOT)}")
    print(f"{'='*72}\n")

    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%SZ")
    results: list[dict] = []

    def emit_live(current_stage: str, stage_status: str,
                   stage_started: datetime,
                   log_path: Optional[Path] = None,
                   log_tail: Optional[list] = None) -> None:
        if dry_run:
            return
        now = datetime.now(timezone.utc)
        write_live({
            "run_id": run_id, "target": target_label,
            "universe": universe, "assets": assets,
            "started_utc": started.isoformat(),
            "now_utc": now.isoformat(),
            "elapsed_s": (now - started).total_seconds(),
            "stages_total": len(walk),
            "stages_done": sum(1 for r in results
                                  if r.get("status") in ("BUILT", "WARNED", "SKIPPED")),
            "current_stage": {
                "name": current_stage,
                "started_utc": stage_started.isoformat(),
                "elapsed_s": (now - stage_started).total_seconds(),
                "status": stage_status,
                "log_path": str(log_path.relative_to(PROJECT_ROOT)) if log_path else None,
                "log_tail": log_tail or [],
            },
            "completed": [
                {"name": r["stage"], "status": r["status"],
                 "elapsed_s": r.get("elapsed_s", 0.0),
                 "gate": r.get("gate", "")}
                for r in results
            ],
        })

    def run_one(stage_name: str) -> dict:
        """Build cmd, invoke producer, run gate, log failure. Returns result dict."""
        spec = dag[stage_name]
        cmd = build_cmd(spec, universe=universe, assets=assets, force=force,
                         workers_override=workers_override)

        if dry_run:
            print(f"  [DRY-RUN] {stage_name}: {' '.join(cmd[1:])}")
            return {"stage": stage_name, "status": "WOULD_BUILD",
                    "cmd": " ".join(cmd[1:]), "elapsed_s": 0.0}

        stage_started = datetime.now(timezone.utc)
        ts = stage_started.strftime("%Y%m%dT%H%M%SZ")
        log_path = LOG_DIR / f"{stage_name}_{ts}.log"
        print(f"  [BUILD] {stage_name} ...  log={log_path.relative_to(PROJECT_ROOT)}",
              flush=True)
        emit_live(stage_name, "BUILDING", stage_started, log_path)

        def _hb(elapsed_s: float, log_tail: list) -> None:
            print(f"  [heartbeat] {stage_name} {elapsed_s:.0f}s elapsed", flush=True)
            emit_live(stage_name, "BUILDING", stage_started, log_path, log_tail=log_tail)

        rc, elapsed, all_present, fresh = invoke_producer(
            spec, cmd, log_path, heartbeat_cb=_hb, heartbeat_seconds=30)

        # STUB detection: rc=0 but no canonical output exists
        if rc == 0 and spec.output_kind != "ephemeral" and not all_present:
            status_v = "STUB"
            gate_msg = "no canonical output exists (smoke-only stub)"
            append_failure(stage_name, "STUB", gate_msg, log_path=str(log_path))
        elif rc == 0:
            ok, gate_msg = run_gate(spec, log_path=log_path)
            status_v = "BUILT" if ok else "GATE_FAILED"
            if not ok:
                append_failure(stage_name, "GATE_FAILED", gate_msg,
                                log_path=str(log_path))
        elif rc == 1 and spec.accepts_rc1:
            status_v = "WARNED"
            ok, gate_msg = run_gate(spec, log_path=log_path)
            if not ok:
                status_v = "GATE_FAILED"
                append_failure(stage_name, "GATE_FAILED", gate_msg,
                                log_path=str(log_path))
        else:
            status_v = "FAILED"
            gate_msg = f"rc={rc}"
            tail = _tail_log_lines(log_path, n=5)
            for ln in tail[-3:]:
                print(f"  [FAIL] {stage_name} | {ln[:180]}", flush=True)
            append_failure(stage_name, "FAILED",
                            f"rc={rc} elapsed={elapsed:.0f}s",
                            log_path=str(log_path))

        state.setdefault("stages", {})[stage_name] = {
            "last_run": datetime.now(timezone.utc).isoformat() + "Z",
            "status": status_v, "elapsed_s": elapsed, "rc": rc,
        }
        save_state(state)
        print(f"  [{status_v}] {stage_name} rc={rc} elapsed={elapsed:.0f}s gate={gate_msg}",
              flush=True)
        emit_live(stage_name, status_v, stage_started, log_path,
                  log_tail=_tail_log_lines(log_path, n=5))
        return {"stage": stage_name, "status": status_v, "rc": rc,
                "elapsed_s": elapsed, "gate": gate_msg,
                "log": str(log_path.relative_to(PROJECT_ROOT))}

    # ───── Execution: serial or wave-parallel ─────
    if parallel:
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        waves = topo_waves(dag, walk)
        print(f"[parallel] {len(walk)} stages -> {len(waves)} waves")
        for wi, w in enumerate(waves, 1):
            print(f"  wave {wi}: {len(w)} stages: {w}")
        for wave_idx, wave in enumerate(waves, 1):
            n_workers = min(max_concurrent_stages, len(wave))
            if any(dag[s].memory_exclusive for s in wave):
                n_workers = 1
            print(f"\n[wave {wave_idx}/{len(waves)}] running {len(wave)} stages "
                  f"with {n_workers} workers...", flush=True)
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futures = {ex.submit(run_one, s): s for s in wave}
                for fut in as_completed(futures):
                    r = fut.result()
                    results.append(r)
    else:
        for stage_name in walk:
            results.append(run_one(stage_name))

    # ───── Summary ─────
    end = datetime.now(timezone.utc)
    n_ok = sum(1 for r in results if r["status"] == "BUILT")
    n_warn = sum(1 for r in results if r["status"] == "WARNED")
    n_fail = sum(1 for r in results if r["status"] in ("FAILED", "GATE_FAILED", "STUB"))
    summary = (f"target={target_label} total={len(results)} "
                f"BUILT={n_ok} WARNED={n_warn} FAILED={n_fail} "
                f"elapsed={(end-started).total_seconds():.0f}s")
    print(f"\n{'='*72}")
    print(f"DONE  {summary}")
    print(f"{'='*72}\n")

    # 2026-05-22: CRIT-5 fix per Opus DAG audit ab3a3e41777c11fde.
    # `pre_train_gate` is DOWNSTREAM of chimera_v51 in the topo, so it never
    # runs when user invokes `--target chimera_v51`. Emit a loud reminder so
    # the validation layer isn't silently skipped. Will be made auto-run in
    # a follow-up; this commit closes the awareness gap.
    if target_label not in ("pre_train_gate", "ALL") and any(
        r["stage"] == "chimera_v51" and r["status"] == "BUILT" for r in results
    ):
        if not any(r["stage"] == "pre_train_gate" for r in results):
            print(f"[!] REMINDER: `pre_train_gate` was NOT run by --target={target_label}.")
            print(f"   Validation layer skipped. Run before training:")
            print(f"     python src/pipeline/refresh.py --target pre_train_gate"
                  f"{f' --universe {universe}' if universe else ''}")
            print()
    if not dry_run:
        write_live({
            "run_id": run_id, "target": target_label,
            "universe": universe, "assets": assets,
            "started_utc": started.isoformat(),
            "now_utc": end.isoformat(),
            "elapsed_s": (end - started).total_seconds(),
            "stages_total": len(walk), "stages_done": n_ok + n_warn,
            "current_stage": None,
            "completed": [
                {"name": r["stage"], "status": r["status"],
                 "elapsed_s": r.get("elapsed_s", 0.0),
                 "gate": r.get("gate", "")}
                for r in results
            ],
            "summary": summary,
        })
    return {"target": target_label, "results": results,
             "started_utc": started.isoformat(),
             "ended_utc": end.isoformat(),
             "summary": summary}


# ─── CLI ────────────────────────────────────────────────────────────────────

def main() -> int:
    # Install signal handlers FIRST so Ctrl-C / SIGTERM during arg-parse or
    # the slow imports below still triggers tree-kill + live-state cleanup.
    _install_signal_handlers()
    try:
        return _main_impl()
    finally:
        _cleanup_on_exit()


def _main_impl() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default=None,
                    help="Target stage to refresh (walks transitive deps).")
    ap.add_argument("--all", action="store_true", dest="refresh_all",
                    help="Refresh EVERY stage in the DAG (topological order).")
    ap.add_argument("--assets", nargs="+", default=None,
                    help="Asset filter (e.g. --assets BTCUSDT ETHUSDT). Passed to "
                         "every per-asset stage in the walk via its canonical "
                         "--assets flag. Cross-section stages accept it as no-op.")
    ap.add_argument("--universe", "--scope", default=None, choices=["u10", "u50", "u100"],
                    dest="universe",
                    help="Universe filter (passed to every universe-aware stage). "
                         "--scope is a deprecated alias for --universe (kept for "
                         "back-compat with run_all_training.py --auto-refresh).")
    ap.add_argument("--force", action="store_true",
                    help="Bypass producer skip-existing; rebuild every stage.")
    ap.add_argument("--workers", type=int, default=None,
                    help="Override --workers for every producer that supports it "
                         "(per-stage max_workers cap applies). "
                         "Default: use per-stage values from config/asset_dag.yaml.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show producer commands without executing.")
    ap.add_argument("--exclude", nargs="+", default=None, metavar="STAGE",
                    help="Stage names to drop from the walk. Excluded stages are "
                         "NOT rebuilt; downstream reads their existing output. "
                         "Warns if an excluded stage has no materialized output.")
    ap.add_argument("--no-deps", action="store_true",
                    help="Run ONLY --target; skip transitive deps. Fails fast if "
                         "any upstream dep is missing on disk.")
    ap.add_argument("--parallel", action="store_true",
                    help="Run independent DAG stages in parallel via wave scheduling. "
                         "memory_exclusive stages auto-singleton-wave.")
    ap.add_argument("--max-concurrent-stages", type=int, default=4,
                    help="Max parallel stages per wave (default 4).")
    ap.add_argument("--status", action="store_true",
                    help="Print freshness table (per-stage last_run + materialized).")
    ap.add_argument("--list", action="store_true",
                    help="List all known assets in the DAG.")
    ap.add_argument("--live", action="store_true",
                    help="Continuously poll _dag_state_live.json and re-render. "
                         "Exits when run finishes or after --live-stale seconds idle.")
    ap.add_argument("--live-once", action="store_true",
                    help="Render --live once and exit.")
    ap.add_argument("--live-interval", type=float, default=2.0,
                    help="--live poll interval seconds (default 2.0).")
    ap.add_argument("--live-stale", type=int, default=300,
                    help="--live exits if state file mtime older than this (default 300s).")
    ap.add_argument("--no-clear", action="store_true",
                    help="--live: don't clear screen between renders.")
    ap.add_argument("--failures", action="store_true",
                    help="Tail logs/refresh/_failures.log.")
    ap.add_argument("--attach", action="store_true",
                    help="Watch an active refresh instead of starting a new one.")
    ap.add_argument("--wait-active", action="store_true",
                    help="If another refresh is running, BLOCK until it finishes.")
    ap.add_argument("--ignore-active", action="store_true",
                    help="Proceed even if another refresh is detected as running. "
                         "RACE possible: only use when targets are disjoint.")
    args = ap.parse_args()

    # ───── Viewer modes (no DAG walk) ─────
    if args.list:
        dag = load_dag()
        print(f"\n{len(dag)} stages in {DAG_PATH.relative_to(PROJECT_ROOT)}:\n")
        for name, spec in sorted(dag.items()):
            deps_s = ", ".join(spec.deps[:3]) + (f"... +{len(spec.deps)-3}"
                                                  if len(spec.deps) > 3 else "")
            print(f"  {name:<28}  per_asset={spec.per_asset!s:<5}  "
                  f"univ={spec.universe_aware!s:<5}  deps=[{deps_s}]")
        return 0

    if args.live or args.live_once:
        return _live_loop(once=args.live_once,
                          interval=args.live_interval,
                          stale_s=args.live_stale,
                          clear_screen=not args.no_clear)

    if args.failures:
        if not FAILURES_LOG.exists():
            print(f"[failures] no failures log at "
                  f"{FAILURES_LOG.relative_to(PROJECT_ROOT)} (none recorded yet)")
            return 0
        try:
            lines = FAILURES_LOG.read_text().splitlines()
        except Exception as e:
            print(f"[failures] read failed: {e}")
            return 2
        print(f"\n{FAILURES_LOG.relative_to(PROJECT_ROOT)}  "
              f"({len(lines)} records, last 50)\n")
        for ln in lines[-50:]:
            print(f"  {ln}")
        return 0

    if args.status:
        status(args.target)
        return 0

    # ───── Active-run detection ─────
    active = detect_active_run()
    if active is not None and not (args.dry_run or args.list or args.failures
                                    or args.live or args.live_once):
        cur = active.get("current_stage", {}) or {}
        active_target = active.get("target")
        active_stage = cur.get("name", "?")
        active_status = cur.get("status", "?")
        active_elapsed_min = (active.get("elapsed_s") or 0) / 60.0

        if args.attach:
            print(f"[refresh] attaching to active run "
                  f"(target={active_target}, stage={active_stage} "
                  f"status={active_status} elapsed={active_elapsed_min:.1f}min)",
                  flush=True)
            return _live_loop(once=False, interval=args.live_interval,
                              stale_s=args.live_stale,
                              clear_screen=not args.no_clear)
        if args.wait_active:
            print(f"[refresh] WAITING for active run; polling every 30s.",
                  flush=True)
            while True:
                time.sleep(30)
                if detect_active_run() is None:
                    print(f"[refresh] active run completed; starting requested refresh",
                          flush=True)
                    break
        elif args.ignore_active:
            print(f"[refresh] WARNING: --ignore-active; proceeding despite active run.",
                  flush=True)
        else:
            print(f"\n[refresh] HALT: another refresh is currently running.")
            print(f"           target={active_target}")
            print(f"           current_stage={active_stage} ({active_status})")
            print(f"           elapsed={active_elapsed_min:.1f}min")
            print(f"\n  Options:")
            print(f"    --attach          watch the active run (read-only)")
            print(f"    --wait-active     block until it finishes, then proceed")
            print(f"    --status          one-shot freshness table")
            print(f"    --ignore-active   PROCEED ANYWAY (race possible)")
            return 2

    if args.refresh_all and args.target:
        ap.error("--all and --target are mutually exclusive")
    if not args.target and not args.refresh_all:
        ap.error("--target required (or pass --all / --list / --status / --live)")

    # ───── Normalize asset format: append USDT if missing ─────
    assets = None
    if args.assets:
        assets = [a.upper() if a.upper().endswith("USDT") else a.upper() + "USDT"
                  for a in args.assets]

    # 2026-05-22 CRIT-1 fix per Opus DAG audit ab3a3e41777c11fde: acquire
    # OS-level exclusive lock before any state writes to close the race
    # window between two near-simultaneous `refresh.py` invocations.
    # --ignore-active still bypasses (user-authorized override).
    if args.ignore_active or args.dry_run:
        result = refresh(args.target,
                         assets=assets, universe=args.universe,
                         force=args.force, dry_run=args.dry_run,
                         workers_override=args.workers,
                         refresh_all=args.refresh_all,
                         exclude=args.exclude,
                         no_deps=args.no_deps,
                         parallel=args.parallel,
                         max_concurrent_stages=args.max_concurrent_stages)
    else:
        try:
            with _RefreshLock():
                result = refresh(args.target,
                                 assets=assets, universe=args.universe,
                                 force=args.force, dry_run=args.dry_run,
                                 workers_override=args.workers,
                                 refresh_all=args.refresh_all,
                                 exclude=args.exclude,
                                 no_deps=args.no_deps,
                                 parallel=args.parallel,
                                 max_concurrent_stages=args.max_concurrent_stages)
        except RuntimeError as e:
            print(f"\n[refresh] HALT: {e}", file=sys.stderr)
            return 2
    n_fail = sum(1 for r in result["results"]
                  if r["status"] in ("FAILED", "GATE_FAILED", "STUB"))
    return 2 if n_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
