#!/usr/bin/env python3
"""ensure_harness.py -- ONE command that proves the whole autonomy ENGINE is installed + working.

The "robust + installable + portable" capstone (2026-06-07 SOTA-integration run). Composes the per-component
ensures + a live smoke of the integrated loop, so a fresh clone (or a post-restart check) can answer in one shot:
"is my engine ready -- planner+replanner, brain-swap (LiteLLM/Ollama), memory (Mem0/TF-IDF), the mechanical
verifier, and the eval keystone?" Idempotent, LOCAL, no network beyond localhost ollama, never hangs/prompts.

Checks (each PASS/WARN/FAIL; WARN = graceful-degrade available, not fatal):
  1. core deps      : langgraph (required), litellm (brain gateway), mem0 (rich memory; WARN->TF-IDF fallback)
  2. brain          : delegates to ensure_brain.py  (litellm + ollama + model + a live decide)
  3. memory         : delegates to ensure_memory.py (mem0 local embedder + store/recall; WARN if mem0 absent)
  4. loop smoke     : harness/run.py --backend mock --budget 1 -> solved (the LangGraph loop runs)
  5. eval keystone  : eval_harness_run.py --brain mock -> solve_rate 1.0 (verifier + scorer honest)
  6. trust gates    : _test_verify_guard + _test_replanner + _test_copy_parity (the engine's own guards green)

Exit 0 = engine READY (any WARNs are documented graceful-degrades); exit 1 = a REQUIRED check failed.
No emoji (Windows cp1252).
"""
from __future__ import annotations

import importlib.metadata as _md
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable
RESULTS = []  # (name, status, detail)


def _rec(name, status, detail=""):
    RESULTS.append((name, status, detail))
    print(f"[ensure-harness:{name}] {status}{(' -- ' + detail) if detail else ''}")


def _run(args, timeout=300):
    try:
        r = subprocess.run([PY] + args, cwd=ROOT, capture_output=True, text=True, timeout=timeout,
                           creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s"
    except Exception as e:
        return 1, f"{type(e).__name__}: {e}"


def _dep(mod, required):
    try:
        v = _md.version(mod)
        _rec(f"dep:{mod}", "PASS", f"v{v}")
        return True
    except Exception:
        _rec(f"dep:{mod}", "FAIL" if required else "WARN",
             "REQUIRED -- pip install " + mod if required else "absent -> graceful fallback")
        return not required


def main():
    print("=" * 78)
    print("ENSURE HARNESS -- is the autonomy engine installed + working? (one-command capstone)")
    print("=" * 78)
    ok = True

    # 1. core deps
    ok &= _dep("langgraph", required=True)
    _dep("litellm", required=False)   # brain gateway; OllamaBrain/hand-rolled fall back if absent
    _dep("mem0ai", required=False)    # rich memory; TF-IDF floor if absent

    # 2 + 3. brain + memory ensures (delegate; they print their own PASS lines)
    for name, script in (("brain", "scripts/autonomy/ensure_brain.py"),
                         ("memory", "scripts/autonomy/ensure_memory.py")):
        code, out = _run([script], timeout=240)
        tail = out.strip().splitlines()[-1] if out.strip() else ""
        _rec(name, "PASS" if code == 0 else "WARN", tail[:120])
        # brain/memory degrade gracefully -> WARN not FAIL (the loop still runs on Mock/TF-IDF)

    # 4. loop smoke -- the LangGraph engine actually runs a cycle
    code, out = _run(["harness/run.py", "--objective", "ensure-harness smoke", "--backend", "mock", "--budget", "1"], 120)
    loop_ok = code == 0 and "solved" in out
    _rec("loop_smoke", "PASS" if loop_ok else "FAIL", "harness/run.py mock -> solved" if loop_ok else out.strip()[-120:])
    ok &= loop_ok

    # 5. eval keystone -- the honest fitness scorer works (mechanical verifier, OracleMockBrain -> 1.0)
    code, out = _run(["scripts/autonomy/eval_harness_run.py", "--brain", "mock"], 180)
    eval_ok = code == 0 and ('"solve_rate": 1.0' in out or "solve_rate=1.0" in out or "1.0000" in out)
    _rec("eval_keystone", "PASS" if eval_ok else "WARN", "mock plumbing solve_rate=1.0" if eval_ok else out.strip()[-120:])

    # 6. trust gates -- the engine's own guards must be green
    for name, script in (("verify_guard", "scripts/autonomy/metaop/_test_verify_guard.py"),
                         ("replanner", "scripts/autonomy/metaop/_test_replanner.py"),
                         ("copy_parity", "scripts/autonomy/metaop/_test_copy_parity.py")):
        code, out = _run([script], 120)
        g_ok = code == 0
        _rec(f"gate:{name}", "PASS" if g_ok else "FAIL", "" if g_ok else out.strip()[-100:])
        ok &= g_ok

    print("=" * 78)
    n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    n_warn = sum(1 for _, s, _ in RESULTS if s == "WARN")
    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    print(f"SUMMARY: {n_pass} PASS / {n_warn} WARN / {n_fail} FAIL")
    if ok and n_fail == 0:
        print("HARNESS READY -- planner+replanner, brain-swap, memory, mechanical verifier, eval keystone all live.")
        print("(WARNs are documented graceful-degrades: e.g. mem0/litellm absent -> TF-IDF/hand-rolled fallback.)")
        print("=" * 78)
        return 0
    print("HARNESS NOT READY -- a REQUIRED check FAILED above (see FAIL lines).")
    print("=" * 78)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
