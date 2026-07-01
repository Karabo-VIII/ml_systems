"""Standardized parallel dispatch for per-pair / per-task pipelines.

Single source of truth for:
  - choosing process vs thread vs serial
  - capping workers to min(workers, len(tasks), MAX)
  - per-task error capture (a worker exception doesn't kill the pool)
  - loud progress logging (heartbeat-friendly)
  - sys.exit(2) on full-fail (no-silent-failure invariant)

Ported from the crypto pipeline framework; behaviour is identical.

Worker function contract:
  worker_fn(*task_args) -> dict
  Result MUST contain a status field (default key='status') whose value
  matches success_value (default='ok') for a successful task. Other fields
  are pretty-printed in the per-task progress line.
"""
from __future__ import annotations

import sys
from concurrent.futures import (
    ProcessPoolExecutor, ThreadPoolExecutor, as_completed,
)
from typing import Any, Callable, Optional

__contract__ = {
    "kind": "framework_helper",
    "stage": "pipeline_dispatch",
    "inputs": {"args": ["tasks", "worker_fn", "workers", "mode"]},
    "outputs": {"side_effects": "loud per-task progress logs; sys.exit(2) on zero ok"},
    "invariants": {
        "worker_exception_capture": True,
        "heartbeat_friendly_logging": True,
        "exit_2_on_zero_ok": True,
    },
    "rationale": "Shared ProcessPool/ThreadPool dispatch across all producers.",
}


VALID_MODES = ("serial", "thread", "process")


def run_per_task(
    tasks: list[tuple],
    worker_fn: Callable[..., dict],
    *,
    workers: int = 1,
    mode: str = "serial",
    stage_name: str = "pipeline",
    cap_workers: int = 16,
    success_field: str = "status",
    success_value: str = "ok",
    exit_on_zero_ok: bool = True,
    progress_summary_keys: Optional[list[str]] = None,
) -> dict:
    """Dispatch worker_fn(*task) over tasks, with parallelism + error capture.

    Each task is a tuple of positional args passed to worker_fn. tasks[i][0]
    is treated as the task id for logging.

    mode: 'serial' | 'thread' (I/O-bound, e.g. HTTP fetch) | 'process'
    (CPU-bound, e.g. bar aggregation). Returns
    {"ok": int, "err": int, "results": list[dict]}.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"run_per_task: mode must be in {VALID_MODES}, got {mode!r}")
    n_total = len(tasks)
    if n_total == 0:
        print(f"[{stage_name}] no tasks dispatched (no-op)", flush=True)
        return {"ok": 0, "err": 0, "results": []}

    workers = max(1, min(workers, n_total, cap_workers))
    executor_cls: Optional[type] = None
    if mode == "thread" and workers > 1:
        executor_cls = ThreadPoolExecutor
    elif mode == "process" and workers > 1:
        executor_cls = ProcessPoolExecutor

    print(f"[{stage_name}] dispatching {n_total} tasks "
          f"(mode={mode}, workers={workers})", flush=True)

    results: list[dict] = []
    n_ok = n_err = 0

    def _format_summary(result: dict) -> str:
        keys = progress_summary_keys
        if keys is None:
            keys = [k for k in result if k not in (success_field, "task")]
        parts = []
        for k in keys[:4]:
            if k in result:
                v = result[k]
                parts.append(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}")
        return " ".join(parts)

    def _record(i: int, task_id: Any, result_or_exc) -> None:
        nonlocal n_ok, n_err
        if isinstance(result_or_exc, BaseException):
            n_err += 1
            print(f"  [{stage_name} {i}/{n_total}] {task_id}  FAIL: "
                  f"{type(result_or_exc).__name__}: {result_or_exc}", flush=True)
            results.append({success_field: "error", "task": task_id,
                           "err": str(result_or_exc)})
            return
        result = result_or_exc
        if not isinstance(result, dict):
            n_err += 1
            print(f"  [{stage_name} {i}/{n_total}] {task_id}  FAIL: "
                  f"worker returned non-dict ({type(result).__name__})", flush=True)
            results.append({success_field: "error", "task": task_id,
                           "err": "non-dict result"})
            return
        status = result.get(success_field)
        summary = _format_summary(result)
        if status == success_value:
            n_ok += 1
            print(f"  [{stage_name} {i}/{n_total}] {task_id}  ok  {summary}", flush=True)
        else:
            n_err += 1
            print(f"  [{stage_name} {i}/{n_total}] {task_id}  {status}  {summary}", flush=True)
        results.append({**result, "task": task_id})

    if executor_cls is None:
        for i, t in enumerate(tasks, 1):
            try:
                r = worker_fn(*t)
            except Exception as e:   # let KeyboardInterrupt / SystemExit bubble
                _record(i, t[0], e)
                continue
            _record(i, t[0], r)
    else:
        with executor_cls(max_workers=workers) as ex:
            futures = {ex.submit(worker_fn, *t): (idx + 1, t[0])
                       for idx, t in enumerate(tasks)}
            for fut in as_completed(futures):
                i, task_id = futures[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    _record(i, task_id, e)
                    continue
                _record(i, task_id, r)

    print(f"[{stage_name}] done: {n_ok} ok / {n_err} err / {n_total} total", flush=True)

    if n_ok == 0 and exit_on_zero_ok and n_total > 0:
        print(f"[{stage_name}] ERROR: zero successes -- exiting non-zero", flush=True)
        sys.exit(2)

    return {"ok": n_ok, "err": n_err, "results": results}
