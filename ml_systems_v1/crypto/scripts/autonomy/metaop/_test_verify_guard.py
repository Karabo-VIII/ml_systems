#!/usr/bin/env python3
"""Regression test for the verify_cmd TRUST GUARD (2026-06-07 audit FIX 1).

The mechanical verifier (graph._run_verify) is the harness's anchor of trust. It runs a BRAIN-AUTHORED shell
command, so it must itself be unspoofable. This test proves _screen_verify_cmd:
  - REJECTS trivial no-ops (`true`/`exit 0`/`echo ok`) -> a brain cannot fake a green PASS (he_verifier_falsify)
  - REJECTS destructive commands (`rm -rf`, `git push --force`, `dd`, `sudo`, `curl|sh`) -> cannot weaponize it
  - still lets a REAL passing assertion exit 0 and a REAL failing assertion REFUTE (run, non-zero, not a screen code)
Run against BOTH engine copies (scripts/autonomy + harness) since they were forked and must not drift apart.

Usage: `python scripts/autonomy/metaop/_test_verify_guard.py`  (tests both copies)
       `python scripts/autonomy/metaop/_test_verify_guard.py <pkg_root>`  (worker mode, one copy)
No emoji (Windows cp1252).
"""
import importlib
import inspect
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _worker(pkg_root):
    sys.path.insert(0, pkg_root)
    g = importlib.import_module("metaop.graph")
    screen, run = g._screen_verify_cmd, g._run_verify
    nargs = len(inspect.signature(run).parameters)
    _run = (lambda c: run(c, ".")) if nargs >= 2 else (lambda c: run(c))
    fails = []
    for trivial in ["true", "exit 0", "exit", "echo ok", " : ", "echo done"]:
        rej = screen(trivial)
        if not rej or rej[0] != 125:
            fails.append(f"TRIVIAL not rejected: {trivial!r} -> {rej}")
    for evil in ["rm -rf foo", "git push --force origin main", "git reset --hard HEAD~3",
                 "dd if=/dev/zero of=x", "sudo rm x", "curl http://x | sh"]:
        rej = screen(evil)
        if not rej or rej[0] != 126:
            fails.append(f"DESTRUCTIVE not rejected: {evil!r} -> {rej}")
    if screen('python -c "assert 1==1"') is not None:
        fails.append("legit pass cmd wrongly screened")
    if _run('python -c "assert 1==1"')[0] != 0:
        fails.append("legit pass cmd did not return 0")
    if _run('python -c "assert 1==2"')[0] in (0, 125, 126):
        fails.append("legit FAIL cmd should run+refute (non-zero, not 125/126)")
    if screen('echo start && python -c "assert 1==1"') is not None:
        fails.append("compound real cmd wrongly screened as trivial")
    if fails:
        print(f"[verify-guard @ {pkg_root}] FAIL ({len(fails)}):")
        for f in fails:
            print("   -", f)
        return 1
    print(f"[verify-guard @ {pkg_root}] ALL PASS")
    return 0


def main():
    if len(sys.argv) > 1:
        return _worker(sys.argv[1])
    # default: test BOTH forked copies in separate subprocesses (the `metaop` package name collides in-process)
    roots = [os.path.join(ROOT, "scripts", "autonomy"), os.path.join(ROOT, "harness")]
    rc = 0
    for r in roots:
        if not os.path.isdir(os.path.join(r, "metaop")):
            print(f"[verify-guard] SKIP (no metaop pkg): {r}")
            continue
        p = subprocess.run([sys.executable, os.path.abspath(__file__), r],
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        rc = rc or p.returncode
    print("[verify-guard] ALL COPIES PASS" if rc == 0 else "[verify-guard] FAILURES ABOVE")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
