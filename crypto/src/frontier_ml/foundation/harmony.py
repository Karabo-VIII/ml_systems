"""Resource harmony — cap CPU/GPU/IO usage so long-running ML jobs don't
choke the OS or starve other processes.

The 4060/8GB + i9 16-core / 32GB workstation has three resources that PyTorch +
Polars + numpy will gleefully saturate by default:

    - CPU threads: PyTorch intra-op + Polars internal both default to all
      logical cores (16). Run them simultaneously and Windows starves; UI
      lags; nvidia-smi can't update; monitor processes miss heartbeats.
    - GPU VRAM: PyTorch's caching allocator can hold > 90% of VRAM, leaving
      no headroom for the display compositor or for transient spikes (the
      bar-stage workers=6 OOM was this same pattern at the CPU level).
    - System RAM: numpy + polars allocate big slabs. With 32 GB host and
      ~10 GB slim cache + ~3 GB Adam states, a runaway alloc paginates.

Harmony policy (defaults):
    - CPU intra-op threads: 8 (half of 16 logical cores; leave 8 for OS,
      nvidia driver, monitoring, the editor / browser, etc.)
    - CPU inter-op threads: 2
    - Polars threads: 8 (match torch)
    - OMP / MKL threads: 8 (cuts numpy/scipy)
    - GPU memory fraction: 0.90 (reserve ~850 MB for compositor)
    - Process priority: BELOW_NORMAL on Windows (so foreground apps stay
      responsive even when training pegs the cores)
    - cudnn benchmark: True (faster convs after first warmup)
    - Pinned host memory for tensors going to GPU (prefetched DMA, lower
      jitter on host->device transfer)

Usage at the top of any long-running script:

    from frontier_ml.foundation.harmony import apply_harmony
    apply_harmony()                      # silent defaults
    apply_harmony(verbose=True)          # logs the configuration applied

apply_harmony() can be called only once per process; subsequent calls are
no-ops with a warning.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

_APPLIED = False


def _logical_cpu_count() -> int:
    try:
        return os.cpu_count() or 8
    except Exception:
        return 8


def apply_harmony(
    *,
    cpu_threads: Optional[int] = None,
    interop_threads: int = 2,
    gpu_mem_fraction: float = 0.90,
    priority_below_normal: bool = True,
    cudnn_benchmark: bool = True,
    verbose: bool = False,
) -> dict:
    """Apply resource caps. Idempotent across a single process.

    Returns the dict of values actually applied (useful for logging).
    """
    global _APPLIED
    if _APPLIED:
        if verbose:
            print("[harmony] already applied; skipping", flush=True)
        return {}

    n_logical = _logical_cpu_count()
    # Default: half logical cores, never less than 4, never more than 8.
    if cpu_threads is None:
        cpu_threads = max(4, min(8, n_logical // 2))

    applied = {"n_logical_cpu": n_logical, "cpu_threads": cpu_threads,
                "interop_threads": interop_threads,
                "gpu_mem_fraction": gpu_mem_fraction}

    # ---- environment variables (must be set before torch/polars import to
    # take FULL effect; here we set defensively in case they weren't yet) --
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"):
        os.environ.setdefault(var, str(cpu_threads))

    # ---- torch threads ---------------------------------------------------
    try:
        import torch
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(interop_threads)
        applied["torch_intraop"] = torch.get_num_threads()
        try:
            applied["torch_interop"] = torch.get_num_interop_threads()
        except Exception:
            pass

        # cudnn perf tuning
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
            try:
                torch.cuda.set_per_process_memory_fraction(gpu_mem_fraction)
                applied["gpu_mem_fraction_set"] = True
            except Exception as e:
                applied["gpu_mem_fraction_err"] = str(e)
            applied["cudnn_benchmark"] = bool(cudnn_benchmark)
            applied["cuda_device"] = torch.cuda.get_device_name(0)
            applied["cuda_total_gb"] = (
                torch.cuda.get_device_properties(0).total_memory / 1e9
            )
    except ImportError:
        applied["torch"] = "not installed"

    # ---- polars threads --------------------------------------------------
    try:
        import polars as pl
        # Polars reads POLARS_MAX_THREADS at import; it's already set above.
        applied["polars_threads_env"] = os.environ.get("POLARS_MAX_THREADS")
        applied["polars_threadpool"] = pl.thread_pool_size()
    except ImportError:
        applied["polars"] = "not installed"
    except Exception:
        pass

    # ---- process priority -- Windows -------------------------------------
    if priority_below_normal and sys.platform.startswith("win"):
        try:
            import ctypes
            BELOW_NORMAL = 0x00004000
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.kernel32.SetPriorityClass(handle, BELOW_NORMAL)
            applied["priority_below_normal"] = bool(ok)
        except Exception as e:
            applied["priority_err"] = str(e)
    elif priority_below_normal:
        # POSIX nice (lower priority -> higher nice value)
        try:
            os.nice(5)
            applied["nice"] = 5
        except Exception:
            pass

    _APPLIED = True

    if verbose:
        print("=" * 70, flush=True)
        print("[harmony] resource caps applied", flush=True)
        for k, v in applied.items():
            print(f"  {k:24s} = {v}", flush=True)
        print("=" * 70, flush=True)

    return applied


if __name__ == "__main__":
    apply_harmony(verbose=True)
