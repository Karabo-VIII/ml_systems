"""scripts/autonomy/cross_pollination_bus.py -- CROSS-POLLINATION BUS for the 3-layer engine.

PURPOSE
-------
The 3 solutioning layers (A=Games/self-play, B=General-WM/time-series, C=Crypto-WM) TEACH EACH
OTHER via transferable Lessons. This module is the mechanical write/read spine that makes it happen
without reinventing storage: it REUSES the skill_library (skill_library.py) for the asset registry
and the JSONL learnings lanes for the raw lesson stream.

STORAGE (two levels, no new files beyond what the autonomy harness already uses):
  - runs/autonomy/cross_pollination.jsonl      : append-only raw lesson stream
  - runs/autonomy/cross_pollination.jsonl.lock : O_CREAT|O_EXCL mutex for concurrent writers
  - runs/autonomy/skill_library/INDEX.json     : skill_library entry per lesson (for digest/search)

LESSON SCHEMA
-------------
  ts (int)           -- Unix timestamp
  lesson_id (str)    -- sha256[:16] of (source_layer + title + body)
  source_layer (str) -- "A", "B", or "C"
  target_layers (list[str]) -- layers that should read this
  category (str)     -- gate | representation | objective_framing | search |
                         robustness | null_model | architecture
  title (str)        -- one-line summary
  body (str)         -- the lesson text (can be multi-sentence)
  transfer_note (str)-- how to APPLY it in the target layer(s)
  evidence_path (str)-- repo-relative path to the artefact that proves it (or "")
  provenance_sha (str)-- git SHA at time of writing (or "")

API
---
  write_lesson(source_layer, target_layers, category, title, body,
               transfer_note="", evidence_path="", provenance_sha="") -> dict
  read_for_layer(target_layer, category=None, k=20) -> list[dict]
  cross_layer_digest(target_layer, k=10) -> str   (prompt-ready)

Seeded with 3 real lessons from this project (see _SEED_LESSONS).
No emoji (Windows cp1252). Concurrent-safe: O_CREAT|O_EXCL file-lock guards the
read-dedup-append critical section (same idiom as git_commit_safe.py).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts" / "autonomy") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "autonomy"))

JSONL_PATH = ROOT / "runs" / "autonomy" / "cross_pollination.jsonl"
_LOCK_PATH = JSONL_PATH.with_suffix(".jsonl.lock")  # <jsonl>.lock

_VALID_LAYERS = {"A", "B", "C"}
_VALID_CATEGORIES = {
    "gate", "representation", "objective_framing", "search",
    "robustness", "null_model", "architecture",
}

# Lock tuning (seconds).
#   _LOCK_STALE_S : a lock is only reclaimed if it is BOTH older than this AND its holder
#                   PID is verifiably dead (a generous window -- a live holder is NEVER stolen).
#   _LOCK_WAIT_S  : safety-net deadline. This is NOT a fail-open: on timeout we RAISE
#                   (TimeoutError), we NEVER write unlocked. The value is large because the lock
#                   is only ever held for a sub-millisecond append; reaching it means a genuine
#                   deadlock (a holder that is neither dead-and-stale nor releasing), which must
#                   surface loudly, not silently lose a lesson.
_LOCK_STALE_S: float = 60.0   # reclaim a lock only if older than this AND holder PID is dead
_LOCK_WAIT_S: float = 60.0    # raise TimeoutError after this (NEVER write unlocked)

# ---------------------------------------------------------------------------
# Seed lessons (real, project-grounded, written once at module level)
# ---------------------------------------------------------------------------

_SEED_LESSONS = [
    {
        "source_layer": "A",
        "target_layers": ["C"],
        "category": "gate",
        "title": "Monotonic champion-gate -> WM compound-promotion gate",
        "body": (
            "The chess self-play engine enforces a monotonic champion gate: a new checkpoint is ONLY "
            "promoted if it WINS against the current champion (Wilson-CI + eval-trust). Without this, "
            "random variance promotes weaker nets (the forgetting axis lesson). The gate is the anti-"
            "self-deception mechanism -- it makes regression impossible by construction."
        ),
        "transfer_note": (
            "For the crypto WM (Layer C): promotion to the 'ship' tier must require strict improvement "
            "on HELD-OUT compound return across 10/10 seeds vs the current best. A WM with ShIC=0.029 "
            "does NOT promote a WM with ShIC=0.028 -- the confidence interval must be non-overlapping. "
            "Wired in: src/strat/battery.py Lens A (n>=15 + jk2>0 + p05>0) is the compound-gate "
            "equivalent; add an explicit 'beats_champion' assertion to the candidate_gate pipeline."
        ),
        "evidence_path": "projects/chess_zero/az/train_robust.py",
        "provenance_sha": "",
    },
    {
        "source_layer": "C",
        "target_layers": ["A"],
        "category": "representation",
        "title": "RevIN temporal-memorization trap -> games encoder normalization",
        "body": (
            "RevIN (instance normalization applied to the input sequence) caused the crypto WM to "
            "memorize temporal distribution SHIFTS instead of learning signal: ShIC collapsed from "
            "0.028 to -0.001 when RevIN was enabled. The net learned 'is the current window higher "
            "than its own mean?' (trivially true in trends) not 'what happens next?'. Invariant: "
            "RevIN DISABLED by default in all V1-V9 models (CLAUDE.md Critical Invariants)."
        ),
        "transfer_note": (
            "For games/Layer A: any per-sample normalization of the observation (layer-norm, instance-"
            "norm, RevIN) over the TEMPORAL dimension of an input sequence risks a similar shortcut -- "
            "the net can exploit 'which half of the sequence is larger' rather than game structure. "
            "Safe: per-channel feature normalization over the training corpus (global stats). "
            "Risky: normalizing each game's SEQUENCE by that game's own stats at inference time."
        ),
        "evidence_path": "src/wm/v1/v1_training/settings.py",
        "provenance_sha": "",
    },
    {
        "source_layer": "A",
        "target_layers": ["C"],
        "category": "objective_framing",
        "title": "Terminal-not-per-step objective -> WM setup/move-onset target",
        "body": (
            "AlphaZero's training signal is the TERMINAL game outcome (win/loss/draw), not a per-step "
            "reward shaped to guide the search. This is the correct framing: the agent learns what "
            "positions are ULTIMATELY valuable, not to optimise a per-ply proxy that might diverge. "
            "Equivalently, self-play with per-move reward shaping degrades -- the forgetting axis "
            "showed that a net optimised on move-count heuristics lost to the terminal-outcome net."
        ),
        "transfer_note": (
            "For the crypto WM (Layer C): the unit of trading is a SETUP across a MULTI-CANDLE MOVE "
            "(MEMORY.md Founding Framing). The correct objective target is the COMPOUND return over the "
            "setup horizon, NOT per-bar IC/return prediction. IC h=1 survives ONLY as a within-WM "
            "diagnostic gate (>0.015), never as the optimisation objective. This maps exactly to "
            "AlphaZero's terminal-only signal: label the SETUP onset, evaluate at SETUP resolution."
        ),
        "evidence_path": "docs/GENERAL_PROBLEM_SOLVING_HARNESS_2026_06_09.md",
        "provenance_sha": "",
    },
]


# ---------------------------------------------------------------------------
# File-lock helpers (O_CREAT|O_EXCL -- same idiom as git_commit_safe.py)
# ---------------------------------------------------------------------------

def _pid_alive(pid: int) -> bool:
    """Return True if a process with `pid` is currently running.

    Conservative by design: when liveness CANNOT be determined, return True (assume alive)
    so we NEVER steal a lock that might still be held.  Only a *confirmed-dead* holder lets
    a stale lock be reclaimed.
    """
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True  # ourselves (e.g. concurrent THREADS share our PID) -- always live
    if os.name == "nt":
        # Windows: query the process table.  tasklist is always available; a CSV row
        # for the PID means it is alive.  Any error -> assume alive (conservative).
        try:
            import subprocess
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
                capture_output=True, text=True, timeout=5,
                creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
            )
            # A live PID prints a quoted CSV row containing the PID; a dead one prints
            # "INFO: No tasks ..." (or nothing).  Require the PID to appear in a CSV cell.
            return f'"{pid}"' in out.stdout
        except Exception:
            return True  # cannot tell -> assume alive (never steal on uncertainty)
    # POSIX: signal 0 probes existence without delivering a signal.
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True   # exists but owned by another user -> alive
    except Exception:
        return True   # cannot tell -> assume alive


def _read_lock_holder(lock_path: Path) -> tuple[int, float]:
    """Return (holder_pid, held_ts) parsed from the lock file.

    On any read failure (incl. Windows delete-pending / held-open PermissionError) return
    a sentinel that makes the lock look LIVE-and-FRESH (pid=-1 unknown, ts=now), so the
    caller waits rather than steals.  Liveness is never inferred from an unreadable lock.
    """
    try:
        parts = lock_path.read_text().split()
        pid = int(parts[0]) if len(parts) > 0 else -1
        held_ts = float(parts[1]) if len(parts) > 1 else time.time()
        return pid, held_ts
    except Exception:
        # Unreadable (held-open on Windows, partial write, etc.) -> treat as live+fresh.
        return -1, time.time()


def _acquire_lock(lock_path: Path, stale_s: float = _LOCK_STALE_S,
                  wait_s: float = _LOCK_WAIT_S) -> bool:
    """Acquire an O_CREAT|O_EXCL file lock.  BLOCKS until acquired; raises on timeout.

    Contract (correctness-critical -- this guards a data-loss append path):
      * Returns True ONLY when the lock is genuinely held by this caller.
      * NEVER returns False and NEVER fails open: a writer either gets the lock or this
        raises TimeoutError.  The caller must therefore NOT write unlocked on a False/raise.
      * A stale lock is reclaimed ONLY if it is BOTH older than `stale_s` AND its holder
        PID is confirmed dead -- a live holder is never stolen (closes the "steal a live
        lock" window).  Same-PID holders (concurrent threads) are always treated as live,
        so threads serialize correctly instead of stealing from each other.

    Windows note: when the lock is in 'delete-pending' limbo (just-unlinked by the releaser)
    or still held-open, os.open raises PermissionError (Errno 13) rather than FileExistsError.
    Both are treated identically as 'contended -- retry', so the previous bug (PermissionError
    escaping uncaught, or being misread as a fresh lock that forced a 15s fail-open) is gone.
    """
    deadline = time.time() + wait_s
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, f"{os.getpid()} {time.time():.6f}\n".encode())
            finally:
                os.close(fd)
            return True
        except (FileExistsError, PermissionError):
            # Lock is contended (held, or Windows delete-pending/held-open).  Decide whether
            # the holder is a confirmed-dead stale lease we may reclaim, else wait.
            pid, held_ts = _read_lock_holder(lock_path)
            age = time.time() - held_ts
            if age > stale_s and not _pid_alive(pid):
                # Confirmed-dead holder with an old lease -> safe to steal.
                try:
                    lock_path.unlink()
                except (FileNotFoundError, PermissionError):
                    pass  # someone else already stole it / still held -- just retry
                continue
            # Live (or not-confirmed-dead) holder: wait and retry, but never write unlocked.
            if time.time() > deadline:
                raise TimeoutError(
                    f"_acquire_lock: could not acquire {lock_path} within {wait_s:.0f}s "
                    f"(holder pid={pid}, lock age={age:.1f}s). Refusing to write unlocked "
                    f"-- a lesson is never silently lost. This indicates a genuine deadlock."
                )
            time.sleep(0.02)


def _release_lock(lock_path: Path) -> None:
    """Release the lock by deleting the lock file.

    On Windows a transient PermissionError (antivirus/indexer briefly holding the handle)
    is retried a few times so the lease is actually freed promptly -- a lingering lock file
    here would force the next writer down the stale-reclaim path unnecessarily.
    """
    for _ in range(5):
        try:
            lock_path.unlink()
            return
        except FileNotFoundError:
            return  # already gone (e.g. reclaimed) -- fine
        except PermissionError:
            time.sleep(0.02)  # Windows: handle briefly held by AV/indexer -- retry
    # Best-effort final attempt; if it still fails the stale-reclaim path will clean up
    # later (the holder PID in the file is ours and will be dead once this process exits).
    try:
        lock_path.unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# I/O helpers (lock-guarded append)
# ---------------------------------------------------------------------------

def _read_all() -> list[dict]:
    """Read all lessons from the JSONL file.  No lock needed for read-only callers."""
    if not JSONL_PATH.exists():
        return []
    lessons = []
    for line in JSONL_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                lessons.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return lessons


def _append_lesson(lesson: dict) -> None:
    """Append one lesson to the JSONL, holding the file lock for the entire
    read-dedup-append critical section so concurrent writers cannot lose each other's lessons.

    Design:
      - Acquire the O_CREAT|O_EXCL lock (stale-reclaim + fail-open timeout).
      - Inside the lock: re-read the file to pick up any lesson written since the caller's
        last _read_all() (the dedup window), check for duplicate, then append a single
        JSON line in 'a' mode (no full-file rewrite needed).
      - Release in finally.

    APPEND mode ('a') is used instead of read-all+rewrite so the write itself is a single
    OS append call; the lock makes the read-then-append atomic as a critical section.
    """
    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    acquired = _acquire_lock(_LOCK_PATH)
    try:
        # Re-read inside the lock to catch any lesson written between our caller's last read
        # and now (closes the TOCTOU window on the dedup check).
        if JSONL_PATH.exists():
            existing_ids = set()
            for line in JSONL_PATH.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        existing_ids.add(json.loads(line).get("lesson_id", ""))
                    except json.JSONDecodeError:
                        pass
        else:
            existing_ids = set()

        if lesson.get("lesson_id", "") in existing_ids:
            # Another concurrent writer beat us -- already committed, nothing to do.
            return

        # Append exactly one JSON line.  'a' mode on Windows is safe for a single write
        # because the lock ensures no concurrent appender.
        with open(JSONL_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(lesson, ensure_ascii=False) + "\n")
    finally:
        if acquired:
            _release_lock(_LOCK_PATH)


def _lesson_id(source_layer: str, title: str, body: str) -> str:
    raw = f"{source_layer}|{title}|{body}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _register_in_skill_library(lesson: dict) -> None:
    """Register the lesson in skill_library so digest/search picks it up.

    Failures are logged as a one-line warning (not silently swallowed) because a failed
    registration means the lesson won't appear in skill_library search -- worth knowing.
    The JSONL is already written and remains the source of truth regardless.
    """
    try:
        import skill_library
        tags = (
            [f"layer_{lesson['source_layer'].lower()}"]
            + [f"for_layer_{t.lower()}" for t in lesson.get("target_layers", [])]
            + [lesson.get("category", "")]
            + ["cross_pollination"]
        )
        tags = [t for t in tags if t]
        skill_library.register(
            name=f"cpbus:{lesson['lesson_id']}",
            kind="harness",
            path=str(JSONL_PATH.relative_to(ROOT)),
            entrypoint=f"read_for_layer(target_layer='{lesson['target_layers'][0] if lesson['target_layers'] else '?'}')",
            signature="(target_layer: str, category=None, k=20) -> list[dict]",
            summary=f"[CP {lesson['source_layer']}->{lesson['target_layers']}] {lesson['title']}",
            tested_on="cross_pollination_bus selftest 2026-06-10",
            provenance_sha=lesson.get("provenance_sha", ""),
            tags=tags,
            added_ts=lesson.get("ts", int(time.time())),
        )
    except Exception as exc:  # skill_library unavailable (import path issue) -- JSONL is still written
        warnings.warn(
            f"[cross_pollination_bus] _register_in_skill_library failed for "
            f"{lesson.get('lesson_id', '?')!r}: {exc}",
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_lesson(
    source_layer: str,
    target_layers: list[str],
    category: str,
    title: str,
    body: str,
    transfer_note: str = "",
    evidence_path: str = "",
    provenance_sha: str = "",
) -> dict:
    """Write a transferable lesson from source_layer to target_layers.

    Parameters
    ----------
    source_layer    : "A", "B", or "C"
    target_layers   : list of layers that should read this lesson, e.g. ["C"]
    category        : one of gate|representation|objective_framing|search|robustness|null_model|architecture
    title           : one-line human-readable summary
    body            : the lesson text
    transfer_note   : how to apply it in the target layer(s)
    evidence_path   : repo-relative path to the supporting artefact (or "")
    provenance_sha  : git SHA at time of writing (or "")
    """
    if source_layer not in _VALID_LAYERS:
        raise ValueError(f"source_layer must be one of {_VALID_LAYERS}, got '{source_layer}'")
    for t in target_layers:
        if t not in _VALID_LAYERS:
            raise ValueError(f"target_layers must be subset of {_VALID_LAYERS}, got '{t}'")
    if category not in _VALID_CATEGORIES:
        raise ValueError(f"category must be one of {_VALID_CATEGORIES}, got '{category}'")

    lesson_id = _lesson_id(source_layer, title, body)

    # Fast-path dedup check (outside the lock) to skip obvious duplicates cheaply.
    # _append_lesson re-checks inside the lock for the concurrent case.
    existing = _read_all()
    if any(l.get("lesson_id") == lesson_id for l in existing):
        print(f"[cross_pollination_bus] already exists: {lesson_id!r} -- skipping")
        return next(l for l in existing if l.get("lesson_id") == lesson_id)

    lesson = {
        "ts": int(time.time()),
        "lesson_id": lesson_id,
        "source_layer": source_layer,
        "target_layers": list(target_layers),
        "category": category,
        "title": title,
        "body": body,
        "transfer_note": transfer_note,
        "evidence_path": evidence_path,
        "provenance_sha": provenance_sha,
    }
    _append_lesson(lesson)
    _register_in_skill_library(lesson)
    print(f"[cross_pollination_bus] written: {source_layer}->{target_layers} [{category}] {title!r}")
    return lesson


def read_for_layer(target_layer: str, category: Optional[str] = None, k: int = 20) -> list[dict]:
    """Return up to k lessons targeted at `target_layer`, optionally filtered by category.

    Returns lessons sorted by ts descending (newest first).
    """
    if target_layer not in _VALID_LAYERS:
        raise ValueError(f"target_layer must be one of {_VALID_LAYERS}, got '{target_layer}'")
    all_lessons = _read_all()
    filtered = [
        l for l in all_lessons
        if target_layer in l.get("target_layers", [])
        and (category is None or l.get("category") == category)
    ]
    filtered.sort(key=lambda l: l.get("ts", 0), reverse=True)
    return filtered[:k]


def cross_layer_digest(target_layer: str, k: int = 10) -> str:
    """Prompt-ready digest of lessons for a target layer. Mirrors skill_library.digest() format."""
    lessons = read_for_layer(target_layer, k=k)
    if not lessons:
        return f"(cross_pollination_bus: no lessons for Layer {target_layer} yet)"

    lines = [f"CROSS-LAYER LESSONS for Layer {target_layer} (read-forward: apply before reinventing):"]
    for l in lessons:
        src = l.get("source_layer", "?")
        tgts = l.get("target_layers", [])
        cat = l.get("category", "")
        lines.append(
            f"\n  [{cat.upper()}] {l.get('title', '')}"
            f"\n    Source: Layer {src} -> Targets: {tgts}"
            f"\n    Lesson: {l.get('body', '')[:200]}"
            f"\n    Apply:  {l.get('transfer_note', '')[:200]}"
            + (f"\n    Evidence: {l['evidence_path']}" if l.get("evidence_path") else "")
        )
    return "\n".join(lines)


def _seed() -> int:
    """Write the 3 seed lessons if not already present. Returns count written."""
    written = 0
    for s in _SEED_LESSONS:
        existing = _read_all()
        lid = _lesson_id(s["source_layer"], s["title"], s["body"])
        if any(l.get("lesson_id") == lid for l in existing):
            continue
        write_lesson(
            source_layer=s["source_layer"],
            target_layers=s["target_layers"],
            category=s["category"],
            title=s["title"],
            body=s["body"],
            transfer_note=s.get("transfer_note", ""),
            evidence_path=s.get("evidence_path", ""),
            provenance_sha=s.get("provenance_sha", ""),
        )
        written += 1
    return written


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest(verbose: bool = True) -> int:
    """Write 3 seed lessons, assert read_for_layer("C") >= 2, assert digest non-empty."""
    failures = 0

    # seed
    seeded = _seed()
    if verbose:
        print(f"[cross_pollination_bus] seeded {seeded} new lesson(s)")

    # read_for_layer("C") must return >=2
    lessons_c = read_for_layer("C")
    if len(lessons_c) >= 2:
        if verbose:
            print(f"  [PASS] read_for_layer('C') -> {len(lessons_c)} lesson(s)")
    else:
        print(f"  [FAIL] read_for_layer('C') -> {len(lessons_c)} (expected >= 2)")
        failures += 1

    # digest must be non-empty
    digest_str = cross_layer_digest("C")
    if digest_str and "no lessons" not in digest_str:
        if verbose:
            print(f"  [PASS] cross_layer_digest('C') non-empty ({len(digest_str)} chars)")
    else:
        print(f"  [FAIL] cross_layer_digest('C') empty or placeholder")
        failures += 1

    # read_for_layer("A") must return >=1 (the RevIN lesson targets A)
    lessons_a = read_for_layer("A")
    if len(lessons_a) >= 1:
        if verbose:
            print(f"  [PASS] read_for_layer('A') -> {len(lessons_a)} lesson(s)")
    else:
        print(f"  [FAIL] read_for_layer('A') -> {len(lessons_a)} (expected >= 1)")
        failures += 1

    if verbose and not failures:
        print("[cross_pollination_bus] PASS: all assertions satisfied")
    return failures


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        prog="cross_pollination_bus",
        description="Cross-layer lesson bus for the 3-layer engine.",
    )
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("selftest", help="Run self-test (seed + assert).")
    sub.add_parser("seed", help="Write seed lessons only.")

    rp = sub.add_parser("read", help="Read lessons for a target layer.")
    rp.add_argument("layer", choices=["A", "B", "C"])
    rp.add_argument("--category", default=None)
    rp.add_argument("--k", type=int, default=20)

    dp = sub.add_parser("digest", help="Print prompt-ready digest for a target layer.")
    dp.add_argument("layer", choices=["A", "B", "C"])
    dp.add_argument("--k", type=int, default=10)

    wp = sub.add_parser("write", help="Write a new lesson.")
    wp.add_argument("--source", required=True, choices=["A", "B", "C"])
    wp.add_argument("--targets", required=True, help="Comma-separated target layers, e.g. A,C")
    wp.add_argument("--category", required=True)
    wp.add_argument("--title", required=True)
    wp.add_argument("--body", required=True)
    wp.add_argument("--transfer-note", dest="transfer_note", default="")
    wp.add_argument("--evidence-path", dest="evidence_path", default="")
    wp.add_argument("--provenance-sha", dest="provenance_sha", default="")

    args = ap.parse_args()

    if args.cmd == "selftest" or args.cmd is None:
        fails = _selftest(verbose=True)
        sys.exit(0 if not fails else 1)

    elif args.cmd == "seed":
        n = _seed()
        print(f"Seeded {n} lesson(s).")

    elif args.cmd == "read":
        lessons = read_for_layer(args.layer, category=args.category, k=args.k)
        print(f"Layer {args.layer}: {len(lessons)} lesson(s)")
        for l in lessons:
            print(f"  [{l['category']}] {l['title']}")

    elif args.cmd == "digest":
        print(cross_layer_digest(args.layer, k=args.k))

    elif args.cmd == "write":
        targets = [t.strip() for t in args.targets.split(",") if t.strip()]
        lesson = write_lesson(
            source_layer=args.source,
            target_layers=targets,
            category=args.category,
            title=args.title,
            body=args.body,
            transfer_note=args.transfer_note,
            evidence_path=args.evidence_path,
            provenance_sha=args.provenance_sha,
        )
        print(f"Written: {lesson['lesson_id']}")
