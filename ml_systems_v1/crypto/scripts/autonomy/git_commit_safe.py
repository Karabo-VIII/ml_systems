"""git_commit_safe.py -- the CANONICAL safe-commit path for concurrent loops (commit-lease, RIGHT layer).

THE BUG IT FIXES (CONFIRMED RECURRING 4x: ecdc6ab 2026-06-07, 74cf331 2026-06-08, 37e652b 2026-06-09 x2):
a concurrent loop's `git add -A && git commit` sweeps ANOTHER committer's already-staged files into ITS
commit (mislabeled, not lost). Root cause = there is ONE git index per repo + no lock spanning the
add->commit sequence. So a pre-COMMIT-hook lease is the WRONG LAYER (the hook fires at commit time, after
the foreign `git add -A` already swept the index). The fix must wrap the ENTIRE add->commit critical section.

THIS TOOL: acquire an O_EXCL lease -> `git add <EXPLICIT PATHS>` (NEVER -A) -> `git commit` -> release.
While the lease is held, any other committer that also uses this tool waits (fail-open after `stale_s`).
Two rules make collisions impossible *between cooperating committers*:
  1. the lease serializes the whole add+commit (not just commit);
  2. explicit paths only -- this committer never stages files it did not name.
A non-cooperating committer (a loop still doing raw `git add -A`) can still collide -- the durable fix is
to route ALL loop commits through here. Fail-open by design (commit-lease ONLY): a stale lease is reclaimed
ONLY when its holder PID is confirmed dead AND the lease is older than `stale_s`; a live holder is never
stolen. If a live holder cannot be acquired within `wait_s` the tool proceeds with a LOUD warning rather than
blocking a commit forever -- a commit is not data-loss (git's own .git/index.lock is the real corruption
backstop), so the lease can never deadlock the project. (Hardened 2026-06-10 to share the bus's lock fixes:
catch Windows PermissionError as contended; real PID-liveness stale-reclaim. The fail-open is the deliberate
inverse of cross_pollination_bus, which RAISES because a lost lesson IS data-loss.)

Usage:
  python scripts/autonomy/git_commit_safe.py --paths src/foo.py docs/bar.md --message "msg" [--repo DIR] [--stale-s 60]
  echo "multi-line msg" | python scripts/autonomy/git_commit_safe.py --paths src/foo.py --message-stdin
Exit: 0 commit made; 1 nothing to commit / git error; 2 lease/usage error. No emoji (cp1252).
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def _git(repo: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True,
                          creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))


def _pid_alive(pid: int) -> bool:
    """Return True if a process with `pid` is currently running.

    Conservative by design (same contract as cross_pollination_bus._pid_alive): when liveness
    CANNOT be determined, return True (assume alive) so a stale-reclaim NEVER steals a lease that
    might still be held. Only a *confirmed-dead* holder lets an old lease be reclaimed.
    """
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True  # ourselves (concurrent THREADS share our PID) -- always live
    if os.name == "nt":
        # Windows: query the process table via tasklist (always present). A CSV row for the PID
        # means alive; "INFO: No tasks ..." means dead. Any error -> assume alive (conservative).
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
                capture_output=True, text=True, timeout=5,
                creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
            )
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
    """Return (holder_pid, held_ts) parsed from the lease file.

    On any read failure (incl. Windows delete-pending / held-open PermissionError, or a partial
    write) return a sentinel that makes the lease look LIVE-and-FRESH (pid=-1 unknown, ts=now) so
    the caller waits rather than steals. Liveness is never inferred from an unreadable lock. This
    closes the old bug where a read failure defaulted held_ts=0 -> age huge -> instant steal of a
    possibly-live lease.
    """
    try:
        parts = lock_path.read_text().split()
        pid = int(parts[0]) if len(parts) > 0 else -1
        held_ts = float(parts[1]) if len(parts) > 1 else time.time()
        return pid, held_ts
    except Exception:
        return -1, time.time()


def acquire(lock_path: Path, stale_s: float, wait_s: float) -> bool:
    """O_CREAT|O_EXCL commit-lease. Returns True if the lease is held by this caller, else False.

    FAIL-OPEN is INTENTIONAL for the commit lease (and ONLY for it):
      * a commit is not data-loss -- git's own .git/index.lock is the real corruption backstop, so
        a serialized-with-warning commit beats a commit blocked forever on a wedged peer;
      * returning False on timeout lets `main()` proceed with a LOUD warning (never silently).
    This is the deliberate inverse of cross_pollination_bus._acquire_lock, which RAISES on timeout
    because losing a lesson IS data-loss. The robustness fixes below are shared with the bus:

      (a) catch PermissionError AND FileExistsError as 'contended -> retry'. On Windows, os.open
          O_CREAT|O_EXCL on a delete-pending or held-open lock raises PermissionError (Errno 13),
          NOT FileExistsError -- the old FileExistsError-only catch let it escape uncaught.
      (b) reclaim a stale lease ONLY if it is BOTH older than `stale_s` AND its holder PID is
          confirmed dead via _pid_alive. A live holder is NEVER stolen (closes the "steal a live
          lease" window); a forged-fresh timestamp no longer matters because liveness, not age
          alone, gates the reclaim; an unreadable lock is treated as live+fresh (never stolen).
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
            # Lease contended (held, or Windows delete-pending/held-open). Reclaim only a
            # confirmed-dead-AND-old lease; otherwise wait. Never steal a live/unreadable lease.
            pid, held_ts = _read_lock_holder(lock_path)
            age = time.time() - held_ts
            if age > stale_s and not _pid_alive(pid):
                try:
                    lock_path.unlink()           # steal the stale lease of a confirmed-dead holder
                except (FileNotFoundError, PermissionError):
                    pass                         # someone else already stole it / still held -- retry
                continue
            if time.time() > deadline:
                return False                     # genuine timeout -> fail-open (main() warns LOUDLY)
            time.sleep(0.25)


def release(lock_path: Path) -> None:
    """Release the lease by deleting the lock file.

    A transient Windows PermissionError (AV/indexer briefly holding the handle) is retried so the
    lease is freed promptly; a lingering lock file would otherwise push the next committer down the
    stale-reclaim path needlessly.
    """
    for _ in range(5):
        try:
            lock_path.unlink()
            return
        except FileNotFoundError:
            return  # already gone (e.g. reclaimed) -- fine
        except PermissionError:
            time.sleep(0.02)  # Windows: handle briefly held by AV/indexer -- retry
    try:
        lock_path.unlink()
    except OSError:
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="git_commit_safe")
    ap.add_argument("--paths", nargs="+", required=True, help="EXPLICIT paths to stage (never -A)")
    ap.add_argument("--message"); ap.add_argument("--message-stdin", action="store_true")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--stale-s", type=float, default=60.0, help="steal a lease older than this (fail-open)")
    ap.add_argument("--wait-s", type=float, default=30.0, help="max wait for a live lease before fail-open")
    a = ap.parse_args(argv)
    repo = str(Path(a.repo).resolve())
    msg = sys.stdin.read() if a.message_stdin else (a.message or "")
    if not msg.strip():
        print("[git_commit_safe] empty commit message", file=sys.stderr); return 2

    lock = Path(repo) / ".git" / "commit_lease.lock"
    got = acquire(lock, a.stale_s, a.wait_s)
    if not got:
        holder_pid, holder_ts = _read_lock_holder(lock)
        print(
            f"[git_commit_safe] WARN: FAIL-OPEN -- could not acquire commit lease within {a.wait_s:.0f}s "
            f"(live holder pid={holder_pid}, lease age={time.time() - holder_ts:.1f}s). Proceeding with the "
            f"add+commit anyway (collision risk accepted: a commit is not data-loss; git index.lock backstops "
            f"corruption). If this recurs, a peer committer is wedged -- investigate pid={holder_pid}.",
            file=sys.stderr,
        )
    try:
        add = _git(repo, "add", "--", *a.paths)
        if add.returncode != 0:
            print(f"[git_commit_safe] git add failed: {add.stderr}", file=sys.stderr); return 1
        # commit ONLY the explicitly-staged paths (pathspec-limited) so a foreign add-A cannot widen this commit
        ci = _git(repo, "commit", "-m", msg, "--", *a.paths)
        if ci.returncode != 0:
            out = (ci.stdout + ci.stderr)
            if "nothing to commit" in out or "no changes added" in out:
                print("[git_commit_safe] nothing to commit", file=sys.stderr); return 1
            print(f"[git_commit_safe] git commit failed: {out}", file=sys.stderr); return 1
        head = _git(repo, "rev-parse", "--short", "HEAD").stdout.strip()
        print(f"[git_commit_safe] committed {head}: {a.paths}")
        return 0
    finally:
        if got:
            release(lock)


def _a_dead_pid() -> int:
    """Spawn-and-reap a throwaway process; return its (now-dead) real PID.

    A real ex-PID is a far better stale-reclaim fixture than a magic constant like 99999 (which
    could coincidentally be alive). On POSIX we reap via wait(); on Windows the handle is closed
    on .wait() and tasklist will report no such PID.
    """
    p = subprocess.Popen([sys.executable, "-c", "pass"],
                         creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
    p.wait()
    # Give the OS a beat to fully tear the PID down (Windows table latency).
    for _ in range(50):
        if not _pid_alive(p.pid):
            break
        time.sleep(0.02)
    return p.pid


def _selftest() -> int:
    """No git needed: validates lease acquire / stale-reclaim / release semantics + a 2-thread
    contention race. Covers the hardened contract:
      - fresh acquire works and writes the lock;
      - a second acquire of a LIVE-held lease never double-acquires (fail-open False after wait_s);
      - a dead-PID + old lease IS reclaimed;
      - a LIVE-PID + old lease is NOT reclaimed (the closed "steal a live lock" window);
      - 2 threads contending: exactly one holds at a time, no double-acquire.
    """
    import tempfile
    import threading
    d = Path(tempfile.mkdtemp())
    lk = d / "commit_lease.lock"

    # 1. fresh acquire
    ok1 = acquire(lk, stale_s=60, wait_s=1)
    held = lk.exists()

    # 2. second acquire of a LIVE-held lease (our PID is in it) must FAIL-OPEN quickly, never steal
    t0 = time.time(); ok2 = acquire(lk, stale_s=60, wait_s=0.5); waited = time.time() - t0
    release(lk); gone = not lk.exists()

    # 3. dead-PID + old lease -> reclaimed immediately
    dead = _a_dead_pid()
    lk.write_text(f"{dead} 1\n"); ok3 = acquire(lk, stale_s=1, wait_s=0.5); release(lk)
    dead_reclaimed = ok3

    # 4. live-PID + OLD lease must NOT be stolen (use OUR pid: _pid_alive(self) is always True).
    #    Forge an ancient ts; acquire must still time out (fail-open False), and our lock survives.
    lk.write_text(f"{os.getpid()} 1\n")
    t1 = time.time(); ok4 = acquire(lk, stale_s=1, wait_s=0.5); waited4 = time.time() - t1
    live_not_stolen = (not ok4) and (waited4 >= 0.4) and lk.exists()
    try:
        lk.unlink()
    except FileNotFoundError:
        pass

    # 5. 2-thread contention: both call acquire; track max concurrent holders. Must never exceed 1.
    holders = {"now": 0, "max": 0}
    hlock = threading.Lock()
    double = {"seen": False}

    def _worker():
        got = acquire(lk, stale_s=60, wait_s=3.0)
        if not got:
            return  # fail-open path: did NOT get the lease, must not enter the critical section
        with hlock:
            holders["now"] += 1
            holders["max"] = max(holders["max"], holders["now"])
            if holders["now"] > 1:
                double["seen"] = True
        time.sleep(0.15)            # hold the lease so the peer genuinely contends
        with hlock:
            holders["now"] -= 1
        release(lk)

    t_a = threading.Thread(target=_worker); t_b = threading.Thread(target=_worker)
    t_a.start(); t_b.start(); t_a.join(); t_b.join()
    no_double_acquire = (not double["seen"]) and (holders["max"] <= 1)
    cleaned = not lk.exists()

    ok = (ok1 and held and (not ok2) and (waited >= 0.4) and gone
          and dead_reclaimed and live_not_stolen and no_double_acquire and cleaned)
    print(
        f"[git_commit_safe selftest] fresh_acquire={ok1} held={held} live_held_failopen={not ok2} "
        f"waited={waited:.2f}s released={gone} dead_pid_reclaimed={dead_reclaimed} "
        f"live_pid_NOT_stolen={live_not_stolen} two_thread_no_double_acquire={no_double_acquire} "
        f"(max_concurrent={holders['max']}) cleaned={cleaned} -> {'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        raise SystemExit(_selftest())
    raise SystemExit(main())
