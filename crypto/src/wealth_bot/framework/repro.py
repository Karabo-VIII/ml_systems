"""Reproducibility helpers for wealth-bot scripts.

All scripts that emit an output JSON should call build_repro_block() and embed
the result as output["reproducibility"]. This ensures cross-script comparisons
are traceable to a specific codebase state, data version, and seed set.

Seed standardization decision (2026-05-25):
  Option chosen: A -- standardize ALL ML scripts on CANONICAL_SEEDS going forward.
  Rationale: option B (keep range(N) + optional canonical override) creates two
  parallel comparison populations, making it impossible to attribute differences to
  methodology vs seed initialization. Option C (new baseline re-run) is a superset
  of A but requires a GPU run; adopt A as the code contract, run the new baseline
  opportunistically.
  Impact: scripts that previously used range(10) (e.g. train_and_audit.py,
  honest_refinement_validator.py) will produce DIFFERENT numbers on re-run.
  Any existing audit JSON produced before 2026-05-25 is labelled
  "pre-standardization-baseline" and must NOT be directly compared against
  post-standardization runs without noting the seed-set change.
"""
from __future__ import annotations

import datetime
import hashlib
import platform
import subprocess
from pathlib import Path
from typing import Optional

CANONICAL_SEEDS: list[int] = [42, 1337, 2024, 7, 13, 21, 99, 100, 314, 271]

# Bump this when the output JSON format changes incompatibly (i.e. a downstream
# consumer reading the block would need to be updated).
SCHEMA_VERSION: str = "2026-05-25"

# Root of the repository (two levels above this file: src/wealth_bot/framework/)
_REPO_ROOT = Path(__file__).resolve().parents[3]


def env_block() -> dict:
    """Python + library versions for the calling process."""
    result: dict[str, str] = {"python": platform.python_version()}
    for lib_name, import_name in [
        ("numpy", "numpy"),
        ("polars", "polars"),
        ("pandas", "pandas"),
        ("lightgbm", "lightgbm"),
        ("scikit-learn", "sklearn"),
        ("scipy", "scipy"),
    ]:
        try:
            import importlib
            mod = importlib.import_module(import_name)
            result[lib_name] = getattr(mod, "__version__", "UNKNOWN")
        except ImportError:
            pass  # omit silently -- not all envs have all libs
    return result


def git_sha(short: bool = True) -> str:
    """Short git SHA of HEAD, or 'UNKNOWN' on failure."""
    try:
        fmt_arg = ["--short"] if short else []
        proc = subprocess.run(
            ["git", "rev-parse"] + fmt_arg + ["HEAD"],
            capture_output=True, text=True, cwd=str(_REPO_ROOT),
        )
        sha = proc.stdout.strip()
        if proc.returncode != 0 or not sha:
            return "UNKNOWN"
        # Mark dirty tree
        dirty_proc = subprocess.run(
            ["git", "diff", "--quiet"],
            capture_output=True, cwd=str(_REPO_ROOT),
        )
        if dirty_proc.returncode != 0:
            return sha + "<dirty>"
        return sha
    except Exception:
        return "UNKNOWN"


def git_branch() -> str:
    """Current git branch name, or 'UNKNOWN' on failure."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, cwd=str(_REPO_ROOT),
        )
        return proc.stdout.strip() if proc.returncode == 0 else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def chimera_mtimes(paths: list[str]) -> dict:
    """{path: ISO8601 mtime UTC} for each chimera parquet path used."""
    result: dict[str, str] = {}
    for p_str in paths:
        p = Path(p_str)
        if p.exists():
            mtime_utc = datetime.datetime.utcfromtimestamp(p.stat().st_mtime).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            # Store relative to repo root when possible for portability
            try:
                key = str(p.relative_to(_REPO_ROOT))
            except ValueError:
                key = str(p)
            result[key] = mtime_utc
        else:
            result[p_str] = "NOT_FOUND"
    return result


def config_sha256(yaml_path: str) -> str:
    """SHA-256 hex digest of the config file content."""
    p = Path(yaml_path)
    if not p.exists():
        return "FILE_NOT_FOUND"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_repro_block(
    *,
    command_line: str,
    config_path: Optional[str] = None,
    chimera_paths: Optional[list[str]] = None,
    seeds: Optional[list[int]] = None,
    feature_set_version: Optional[str] = None,
    extra: Optional[dict] = None,
) -> dict:
    """Return the full reproducibility block to embed at the top of output JSON.

    Parameters
    ----------
    command_line:
        The full command used to run the script, e.g. " ".join(sys.argv).
    config_path:
        Path to the YAML config file (for sha256 fingerprinting).
    chimera_paths:
        List of chimera parquet paths consumed by this run (for mtime tracking).
    seeds:
        Seed list used. Defaults to CANONICAL_SEEDS.
    feature_set_version:
        Human label for the feature set, e.g. "f127", "f133".
    extra:
        Any additional key/value pairs to include verbatim.
    """
    block: dict = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha(short=True),
        "git_branch": git_branch(),
        "env": env_block(),
        "seeds": seeds if seeds is not None else CANONICAL_SEEDS,
        "command_line": command_line,
    }
    if config_path is not None:
        block["config_path"] = config_path
        block["config_sha256"] = config_sha256(config_path)
    if chimera_paths is not None:
        block["chimera_mtime_utc"] = chimera_mtimes(chimera_paths)
    if feature_set_version is not None:
        block["feature_set_version"] = feature_set_version
    if extra:
        block.update(extra)
    return block
