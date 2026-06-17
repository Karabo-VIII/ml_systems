"""Harness CONFIG -- the single place the harness binds to a filesystem WORKSPACE.

This module replaces the original metaop's hardcoded `ROOT = Path(__file__).parents[3]` (which pinned the
package to one specific crypto repo). Here the harness is project-AGNOSTIC: a *workspace* is wherever you point
it, and all run artifacts (traces, learnings, durable checkpoints, scratch snippets) live under it -- never
inside the package, never inside someone else's repo.

Resolution order for the workspace root:
  1. an explicit `workspace=` argument passed by the caller (run.py / your code)
  2. the HARNESS_WORKSPACE environment variable
  3. ./.harness_runs under the current working directory (sensible default)

The *build cwd* (where workers actually run shell/python and write artifacts) is a SEPARATE knob (`--cwd` /
HARNESS_CWD), defaulting to the current working directory. Keeping "where artifacts are built" distinct from
"where the harness logs its own bookkeeping" is what lets this drive a build in ANY domain. No emoji (cp1252).
"""
from __future__ import annotations

import os
from pathlib import Path


def workspace_root(workspace: str | None = None) -> Path:
    """Resolve + create the harness workspace root (bookkeeping: traces / learnings / checkpoints)."""
    base = workspace or os.environ.get("HARNESS_WORKSPACE") or os.path.join(os.getcwd(), ".harness_runs")
    p = Path(base).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_cwd(cwd: str | None = None) -> Path:
    """Resolve the directory where WORKERS run their tools and write build artifacts (the target project)."""
    base = cwd or os.environ.get("HARNESS_CWD") or os.getcwd()
    return Path(base).resolve()


def trace_dir(workspace: str | None = None) -> Path:
    d = workspace_root(workspace) / "traces"
    d.mkdir(parents=True, exist_ok=True)
    return d


def learnings_dir(workspace: str | None = None) -> Path:
    d = workspace_root(workspace) / "learnings"
    d.mkdir(parents=True, exist_ok=True)
    return d


def checkpoint_db(workspace: str | None = None, thread: str = "t1") -> Path:
    return workspace_root(workspace) / f"metaop_{thread}.db"


def scratch_dir(workspace: str | None = None) -> Path:
    d = workspace_root(workspace) / "scratch"
    d.mkdir(parents=True, exist_ok=True)
    return d
