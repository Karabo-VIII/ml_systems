"""Bar fabric orchestrator — builds all bar types for the pipeline T1 stage.

Wraps the existing builders under src/pipeline/bars/ and runs them
sequentially, writing into data/processed/bars/<bartype>/. Replaces the
no-op `bar_fabric.py` (an inventory reader) as the pipeline stage cmd.

Built bar types (one subfolder each under data/processed/bars/):
    dib/          — Dollar Imbalance Bars
    runs_tick/    — runs by tick count
    runs_volume/  — runs by volume
    range/        — fixed-range bars
    adaptive_vol/ — adaptive vol bars

Usage:
    python src/pipeline/build_bars.py                       # all bar types, u10
    python src/pipeline/build_bars.py --bartypes dib runs   # subset
    python src/pipeline/build_bars.py --asset BTC           # single
"""

from __future__ import annotations

# CDAP contract — declared after __future__ per PEP-236.
__contract__ = {
    "kind": "pipeline_stage",
    "stage": "bar_fabric",
    "inputs": {
        "args": ["--bartypes", "--asset", "--universe {u10|u50|u100}"],
        "upstream": "data/raw/<SYM>USDT/aggTrades/*.parquet",
    },
    "outputs": {
        "files": "data/processed/bars/<bartype>/<sym>usdt_<bartype>_*.parquet",
        "bartypes": ["dib", "runs_tick", "runs_volume", "range", "adaptive_vol"],
    },
    "invariants": {
        "asset_set_eq": "downstream:hawkes_branching",
        "atomic_write": True,
        "coverage_report_at_end": True,
    },
}

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON = sys.executable

BAR_BUILDERS = {
    "dib":          "src/pipeline/bars/dib_bars_fast.py",
    "runs":         "src/pipeline/bars/runs_bars.py",
    "range":        "src/pipeline/bars/range_bars_fast.py",
    "adaptive_vol": "src/pipeline/bars/adaptive_vol_bars.py",
}

DEFAULT_U10_ASSETS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
]


def run_one(bartype: str, script: str, assets: list[str],
            extra_args: list[str] | None = None) -> tuple[bool, float]:
    """Invoke a bar builder serially (workers=1 path). Streams stdout live so
    run_pipeline's heartbeat can tail meaningful progress."""
    cmd = [PYTHON, script, "--assets"] + assets + list(extra_args or [])
    print(f"  [START] {bartype}: {' '.join(cmd[:3])} ...", flush=True)
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=3600)
        ok = proc.returncode == 0
        if not ok:
            print(f"  [FAIL] {bartype}: exit {proc.returncode}", flush=True)
        else:
            print(f"  [OK] {bartype}: {time.time() - t0:.1f}s", flush=True)
        return ok, time.time() - t0
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {bartype}: > 1h", flush=True)
        return False, time.time() - t0


def _read_last_log_line(log_path: Path, max_chars: int = 90) -> str:
    """Tail the last non-empty line of a child's log file (truncated)."""
    try:
        if not log_path.exists() or log_path.stat().st_size == 0:
            return ""
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 2048))
            chunk = f.read().decode("utf-8", errors="replace")
        for ln in reversed(chunk.splitlines()):
            s = ln.strip()
            if s:
                return s[:max_chars]
    except Exception:
        return ""
    return ""


def _count_bartype_files(bartype: str) -> int:
    """Live count of parquet files produced for this bartype so far."""
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "src" / "pipeline"))
        import layout as _layout
        d = _layout.bars_dir(bartype if bartype != "runs" else "runs_tick")
        if d.exists():
            n = sum(1 for _ in d.glob("*.parquet"))
            # `runs` bartype splits into runs_tick + runs_volume; sum both
            if bartype == "runs":
                d2 = _layout.bars_dir("runs_volume")
                if d2.exists():
                    n += sum(1 for _ in d2.glob("*.parquet"))
            return n
    except Exception:
        pass
    return 0


def run_parallel_bartypes(args_bartypes: list, assets: list[str],
                           workers: int,
                           heartbeat_seconds: int = 15,
                           extra_args: list[str] | None = None) -> tuple[int, int, float, dict]:
    """Run bartypes concurrently as separate subprocesses.

    Each child gets a thread budget via POLARS_MAX_THREADS / RAYON_NUM_THREADS
    so the cumulative thread count across `workers` children doesn't
    oversubscribe the CPU. Mirrors the make_dataset.py thread-budgeting pattern.

    Each child's stdout/stderr is tee'd to logs/build_bars/<bartype>.log so
    log files don't interleave. A heartbeat thread polls each running child
    every `heartbeat_seconds` and prints a per-bartype rollup of:
      - elapsed wall-clock
      - parquet files written so far (vs len(assets) target)
      - tail of the child's log (last non-empty line)
    """
    import os
    import threading
    cpu = os.cpu_count() or 8
    polars_threads = max(2, cpu // max(workers, 1))
    print(f"  Worker thread budget: cpu={cpu} workers={workers} "
          f"-> polars_threads/worker={polars_threads} (total={workers * polars_threads})",
          flush=True)

    log_dir = PROJECT_ROOT / "logs" / "build_bars"
    log_dir.mkdir(parents=True, exist_ok=True)
    n_assets = len(assets)

    # Slot pool: cap concurrent processes at `workers`.
    pending = list(args_bartypes)        # FIFO queue of bartypes still to spawn
    running: list = []                    # (bartype, Popen, log_file, t0, log_path)
    results: dict = {}                    # bartype -> (ok, elapsed)
    started_total = time.time()
    running_lock = threading.Lock()
    stop_hb = threading.Event()

    def _heartbeat():
        # Periodically prints per-bartype progress while children are alive.
        # Reads the child's log + counts output files for live progress.
        while not stop_hb.wait(heartbeat_seconds):
            with running_lock:
                snapshot = list(running)        # shallow copy
            if not snapshot:
                continue
            elapsed_total = time.time() - started_total
            print(f"  [HEARTBEAT] elapsed {elapsed_total:.1f}s · "
                  f"{len(snapshot)}/{len(args_bartypes)} bartypes running, "
                  f"{len(results)} done", flush=True)
            for (bt, proc, _lf, t0, log_path) in snapshot:
                bt_elapsed = time.time() - t0
                n_files = _count_bartype_files(bt)
                tail = _read_last_log_line(log_path)
                pct = (n_files / n_assets * 100.0) if n_assets > 0 else 0.0
                tail_part = f" :: {tail}" if tail else ""
                print(f"    [{bt:<14}] {bt_elapsed:>6.1f}s · "
                      f"{n_files:>3}/{n_assets} files ({pct:>5.1f}%)"
                      f"{tail_part}", flush=True)

    def _spawn(bartype: str):
        script = BAR_BUILDERS[bartype]
        if not (PROJECT_ROOT / script).exists():
            print(f"  [SKIP] {bartype}: builder script missing ({script})", flush=True)
            results[bartype] = (False, 0.0)
            return None
        cmd = [PYTHON, script, "--assets"] + assets + list(extra_args or [])
        env = os.environ.copy()
        env["POLARS_MAX_THREADS"] = str(polars_threads)
        env["RAYON_NUM_THREADS"]  = str(polars_threads)
        env["OMP_NUM_THREADS"]    = str(polars_threads)
        # Force unbuffered stdout in the child so the log file gets live updates
        # (otherwise tqdm / print can buffer and the heartbeat shows nothing).
        env["PYTHONUNBUFFERED"]   = "1"
        log_path = log_dir / f"{bartype}.log"
        log_f = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                                env=env, cwd=str(PROJECT_ROOT))
        print(f"  [SPAWN] {bartype}: pid={proc.pid}, log={log_path.relative_to(PROJECT_ROOT)}",
              flush=True)
        return (bartype, proc, log_f, time.time(), log_path)

    def _drain_one():
        """Wait for any one child to finish; record result; return its bartype."""
        while True:
            with running_lock:
                snapshot = list(running)
            for idx, (bt, proc, log_f, t0, log_path) in enumerate(snapshot):
                rc = proc.poll()
                if rc is not None:
                    log_f.close()
                    elapsed = time.time() - t0
                    ok = rc == 0
                    marker = "OK" if ok else f"FAIL(exit={rc})"
                    n_files = _count_bartype_files(bt)
                    print(f"  [{marker}] {bt}: {elapsed:.1f}s · "
                          f"{n_files}/{n_assets} files written", flush=True)
                    results[bt] = (ok, elapsed)
                    with running_lock:
                        # Find by pid since list may have shifted
                        for j, item in enumerate(running):
                            if item[1].pid == proc.pid:
                                running.pop(j)
                                break
                    return bt
            time.sleep(0.5)

    # Start the heartbeat
    hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    hb_thread.start()

    try:
        # Spawn loop: start up to `workers` at a time
        while pending or running:
            while pending and len(running) < workers:
                bt = pending.pop(0)
                spawned = _spawn(bt)
                if spawned is not None:
                    with running_lock:
                        running.append(spawned)
            if running:
                _drain_one()
    finally:
        stop_hb.set()
        hb_thread.join(timeout=2.0)

    n_ok = sum(1 for _, (ok, _) in results.items() if ok)
    n_fail = sum(1 for _, (ok, _) in results.items() if not ok)
    total_wall = time.time() - started_total
    return n_ok, n_fail, total_wall, results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bartypes", nargs="+", default=list(BAR_BUILDERS.keys()),
                    choices=list(BAR_BUILDERS.keys()),
                    help="Bar types to build (default: all)")
    ap.add_argument("--asset", default=None,
                    help="Single asset (e.g. BTC). Default: --universe u10.")
    ap.add_argument("--universe", default=None,
                    choices=["u10", "u50", "u100"],
                    help="Build for a universe (overrides --asset).")
    ap.add_argument("--workers", type=int, default=1,
                    help="Concurrent BARTYPES (not assets). Default 1 = sequential. "
                         "Cap = len(--bartypes); meaningful values 1..5. Each child "
                         "gets a polars-thread budget = cpu_count // workers so total "
                         "thread count stays under CPU capacity. Each bartype's log "
                         "lands at logs/build_bars/<bartype>.log to avoid interleaving. "
                         "NOTE: build_bars parallelism is at the bartype level — each "
                         "bartype iterates assets serially internally. To parallelize "
                         "asset processing within a bartype, that's a separate refactor "
                         "of the underlying frontier/pipeline/*_bars*.py scripts.")
    ap.add_argument("--heartbeat-seconds", type=int, default=15,
                    help="Heartbeat cadence in concurrent (workers>1) mode. Prints "
                         "per-bartype elapsed + file count + log tail every N seconds. "
                         "Default 15. Ignored when workers=1 (live stdout streamed instead).")
    ap.add_argument("--force", action="store_true",
                    help="Force fresh rebuild: delete prior dated bar files in "
                         "data/processed/bars/<bartype>/ for the resolved universe "
                         "before invoking bartype scripts, and pass --force to each "
                         "bartype script so they bypass per-asset freshness skip "
                         "(default behavior is skip-if-output-newer-than-raw).")
    args = ap.parse_args()

    # Cap workers to len(bartypes) -- can't have more concurrent bartypes than
    # there are bartypes. If the user requested more, surface the cap LOUDLY so
    # they know why the announced value differs from what they passed.
    requested_workers = args.workers
    args.workers = max(1, min(args.workers, len(args.bartypes)))
    if requested_workers > args.workers:
        print(f"  [WARN] --workers {requested_workers} requested but only "
              f"{len(args.bartypes)} bartypes selected ({args.bartypes}); "
              f"capped to workers={args.workers}.", flush=True)
        print(f"         To get more parallelism, either:", flush=True)
        print(f"           (a) select more bartypes (current cap = "
              f"{len(BAR_BUILDERS)} bartypes total)", flush=True)
        print(f"           (b) parallelize asset-level work in the underlying "
              f"frontier/pipeline/*_bars*.py scripts (separate refactor)", flush=True)

    if args.universe:
        # Load asset list from config/universes/<u10|u50|u100>.yaml
        import yaml as _yaml
        univ_path = PROJECT_ROOT / "config" / "universes" / f"{args.universe}.yaml"
        with open(univ_path) as _f:
            spec = _yaml.safe_load(_f)
        # Both u10 and u50 use `assets: [{symbol: BTCUSDT, ...}]`; u100 uses
        # `extra_assets:` + `inherit_from: u50`. Resolve recursively.
        def _resolve(s):
            out = []
            if "inherit_from" in s and s["inherit_from"]:
                parent_path = PROJECT_ROOT / "config" / "universes" / f"{s['inherit_from']}.yaml"
                with open(parent_path) as _g:
                    out.extend(_resolve(_yaml.safe_load(_g)))
            for a in (s.get("assets") or []):
                out.append(a["symbol"])
            for a in (s.get("extra_assets") or []):
                out.append(a["symbol"])
            return out
        excluded = set(spec.get("excluded_assets") or [])
        assets = [a for a in _resolve(spec) if a not in excluded]
        # Dedup preserving order
        seen = set()
        assets = [a for a in assets if not (a in seen or seen.add(a))]
        print(f"  [build_bars] universe={args.universe}: {len(assets)} assets",
              flush=True)
    elif args.asset:
        sym = args.asset.upper()
        if not sym.endswith("USDT"):
            sym += "USDT"
        assets = [sym]
    else:
        assets = DEFAULT_U10_ASSETS

    # @browser B1: --force LOUD; delete prior dated bar files before rebuild.
    if args.force:
        try:
            if str(Path(__file__).resolve().parent) not in sys.path:
                sys.path.insert(0, str(Path(__file__).resolve().parent))
            import layout as _layout
            n_deleted = 0
            for bartype in args.bartypes:
                # bartype scripts use both "runs" (orchestrator) and "runs_tick"/"runs_volume" (output)
                bartype_dirs = [bartype]
                if bartype == "runs":
                    bartype_dirs = ["runs_tick", "runs_volume"]
                for bt_dir in bartype_dirs:
                    try:
                        d = _layout.bars_dir(bt_dir)
                    except Exception:
                        continue
                    if not d.exists():
                        continue
                    for sym in assets:
                        sym_l = sym.lower().replace("USDT", "")
                        for old in d.glob(f"{sym_l}_*.parquet"):
                            try:
                                old.unlink()
                                n_deleted += 1
                            except Exception:
                                pass
                        # Also handle the legacy uppercase pattern (BTCUSDT_dib_2025.parquet)
                        for old in d.glob(f"{sym}_*.parquet"):
                            try:
                                old.unlink()
                                n_deleted += 1
                            except Exception:
                                pass
            print(f"[FORCE] deleted {n_deleted} prior bar snapshots before rebuild", flush=True)
        except Exception as e:
            print(f"[FORCE] WARN deletion failed ({type(e).__name__}: {e}); proceeding",
                  flush=True)

    print(f"\n{'='*70}")
    print(f"BUILD BARS  bartypes={args.bartypes}  assets={len(assets)}  "
          f"workers={args.workers}  force={args.force}")
    print(f"{'='*70}\n")

    # Propagate --force to the child bartype scripts so they bypass per-asset
    # freshness skip (the orchestrator deletes prior files AND the children must
    # not skip-if-fresh, else a force-delete leaves nothing rebuilt).
    child_extra = ["--force"] if args.force else []
    if args.workers > 1:
        # Concurrent path: bartypes run in parallel as separate subprocesses
        # (asset-iteration within each bartype is still serial -- the underlying
        # bartype scripts don't accept --workers themselves).
        n_ok, n_fail, total_t, _ = run_parallel_bartypes(
            args.bartypes, assets, workers=args.workers,
            heartbeat_seconds=args.heartbeat_seconds, extra_args=child_extra)
    else:
        # Sequential path (original behaviour, preserved for run_pipeline
        # heartbeat compatibility -- live stdout streaming).
        n_ok = n_fail = 0
        total_t = 0.0
        n_total = len(args.bartypes)
        for i, bartype in enumerate(args.bartypes, start=1):
            script = BAR_BUILDERS[bartype]
            pct = 100.0 * (i - 1) / n_total
            print(f"\n  [Bar {i}/{n_total}] {bartype} ({pct:.0f}% suite complete)", flush=True)
            if not (PROJECT_ROOT / script).exists():
                print(f"  [SKIP] {bartype}: builder script missing ({script})", flush=True)
                n_fail += 1
                continue
            ok, t = run_one(bartype, script, assets, extra_args=child_extra)
            total_t += t
            if ok:
                n_ok += 1
            else:
                n_fail += 1

    print(f"\n{'='*70}")
    print(f"DONE: {n_ok} ok / {n_fail} fail / {total_t:.1f}s "
          f"{'wall-clock' if args.workers > 1 else 'total'}")
    print(f"{'='*70}")

    # Per-bartype, per-asset coverage report (uniform across pipeline stages).
    try:
        if str(Path(__file__).resolve().parent) not in sys.path:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
        from coverage_report import print_coverage_report
        import layout as _layout
        from collections import defaultdict
        per_bar_ok: dict = defaultdict(set)

        # Map orchestrator bartype name -> list of (canonical_layout_dir, legacy_dir)
        # tuples to scan. The `runs` umbrella expands to two sub-dirs; legacy
        # vol_runs -> runs_vol on disk; layout expects runs_volume.
        BARTYPE_DIRS = {
            "dib":          ["dib"],
            "range":        ["range"],
            "adaptive_vol": ["adaptive_vol"],
            "runs":         ["runs_tick", "runs_volume", "runs_vol"],
            "runs_tick":    ["runs_tick"],
            "runs_volume":  ["runs_volume", "runs_vol"],
        }
        for bartype in args.bartypes:
            dirs_to_scan = BARTYPE_DIRS.get(bartype, [bartype])
            for sub in dirs_to_scan:
                try:
                    d = _layout.bars_dir(sub) if sub in _layout.BAR_TYPES else _layout.DIR_BARS / sub
                except Exception:
                    d = _layout.DIR_BARS / sub
                if not d.exists():
                    continue
                for f in d.glob("*.parquet"):
                    stem = f.stem.lower()
                    if "usdt_" in stem:
                        sym = stem.split("usdt_", 1)[0].upper() + "USDT"
                        per_bar_ok[bartype].add(sym)
        # Aggregate: asset is OK if it produced files for ALL requested bartypes.
        # For `runs` umbrella, presence in EITHER tick or vol counts (already
        # collapsed by the scan above).
        ok_assets = set(a.upper() for a in assets)
        for bt in args.bartypes:
            ok_assets &= per_bar_ok.get(bt, set())
        # Per-bartype summary
        bartype_lines = []
        for bt in args.bartypes:
            n_present = len(per_bar_ok.get(bt, set()) & set(a.upper() for a in assets))
            bartype_lines.append(f"  {bt}: {n_present}/{len(assets)} assets")
        print_coverage_report(
            stage_name="bar_fabric",
            universe=args.universe,
            expected_assets=assets,
            ok_assets=ok_assets,
            err_assets=set(),
            extra_lines=["Per-bartype:"] + bartype_lines,
        )
    except Exception as e:
        print(f"[coverage] WARN: report generation failed: {type(e).__name__}: {e}",
              flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
