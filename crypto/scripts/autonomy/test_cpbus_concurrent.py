"""test_cpbus_concurrent.py -- concurrent-write correctness test for cross_pollination_bus.

This is a DATA-LOSS regression test.  The bus serializes concurrent writers behind an
O_CREAT|O_EXCL file lock; if that lock is not truly robust, a lesson can be silently lost
(overwritten / dropped) under contention.  A single passing run proves NOTHING because the
race is intermittent (it lost ~1-in-10 lessons on a fraction of runs), so this test:

  * runs the 10-way concurrent write 5 TIMES IN A ROW and requires ALL 10 to land
    EVERY single time (50/50 total) -- one loss in any round fails the whole test;
  * runs a 20-way concurrent write once and requires all 20 to land;
  * exits 1 on ANY loss in ANY round (the exit code reflects pass/fail).

It also demonstrates the pre-fix race by temporarily disabling the lock (bypass mode) so the
failure mode this guards against is visible.

Usage:
    python scripts/autonomy/test_cpbus_concurrent.py
Exit 0 = every round landed 100% (fix is working).
Exit 1 = lesson loss detected in any round (bug present).

No emoji (cp1252 safe).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts" / "autonomy") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "autonomy"))

import cross_pollination_bus as bus

# Concurrency knobs
N = 10            # writers per standard round
ROUNDS = 5        # standard rounds run back-to-back (must ALL land 100%)
N_BIG = 20        # writers in the single high-contention round

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lesson_kwargs(i: int) -> dict:
    """Return a unique lesson kwarg dict for writer index i."""
    return dict(
        source_layer="B",
        target_layers=["C"],
        category="robustness",
        title=f"concurrent_test_lesson_{i:03d}",
        body=f"This is the body of concurrent test lesson {i:03d} -- unique sentinel.",
        transfer_note="",
        evidence_path="",
        provenance_sha="",
    )


def _count_landed(jsonl: Path) -> set[str]:
    """Return the set of distinct lesson_ids that actually landed in the JSONL."""
    landed: set[str] = set()
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    landed.add(json.loads(line).get("lesson_id", ""))
                except json.JSONDecodeError:
                    pass
    return landed


def _run_fixed(tmp_jsonl: Path, n_writers: int) -> tuple[int, int, list[str]]:
    """Run `n_writers` concurrent writers against the FIXED bus (lock enabled).

    Returns (n_returned_distinct, n_landed_distinct, errors).
    Each writer writes a DISTINCT lesson; with a correct lock ALL must land.
    """
    # Point bus at a fresh temp file (save/restore the module globals).
    original_path = bus.JSONL_PATH
    original_lock = bus._LOCK_PATH
    bus.JSONL_PATH = tmp_jsonl
    bus._LOCK_PATH = tmp_jsonl.with_suffix(".jsonl.lock")

    errors: list[str] = []
    written_ids: list[str] = []
    lock = threading.Lock()

    def worker(i: int) -> None:
        try:
            kwargs = _make_lesson_kwargs(i)
            lesson = bus.write_lesson(**kwargs)
            with lock:
                written_ids.append(lesson["lesson_id"])
        except Exception as exc:  # a raised TimeoutError counts as a failure, not a silent loss
            with lock:
                errors.append(f"worker {i}: {type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n_writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)  # generous: the fix blocks rather than fail-opens

    landed = _count_landed(tmp_jsonl)

    bus.JSONL_PATH = original_path
    bus._LOCK_PATH = original_lock

    return len(set(written_ids)), len(landed), errors


def _run_unfixed(tmp_jsonl: Path) -> tuple[int, int]:
    """Run N concurrent writers using the OLD racy read-modify-write (no lock).

    This uses a patched _append_lesson that bypasses the lock to demonstrate the
    pre-fix race condition.  Returns (n_write_attempts, n_in_file).
    """
    original_path = bus.JSONL_PATH
    original_lock = bus._LOCK_PATH
    bus.JSONL_PATH = tmp_jsonl
    bus._LOCK_PATH = tmp_jsonl.with_suffix(".jsonl.lock")

    # Monkey-patch _append_lesson to the OLD racy implementation
    original_append = bus._append_lesson

    def _racy_append(lesson: dict) -> None:
        """Old implementation: read-all -> append -> write-whole-file (no lock)."""
        bus.JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        if bus.JSONL_PATH.exists():
            lines = bus.JSONL_PATH.read_text(encoding="utf-8").splitlines()
        # Inject a short sleep to widen the race window
        time.sleep(0.01)
        lines.append(json.dumps(lesson, ensure_ascii=False))
        content = "\n".join(lines) + "\n"
        tmp = bus.JSONL_PATH.with_suffix(".tmp")
        tmp.write_text(content, encoding="utf-8")
        try:
            os.replace(str(tmp), str(bus.JSONL_PATH))
        except Exception:
            tmp.rename(bus.JSONL_PATH)

    bus._append_lesson = _racy_append  # type: ignore[method-assign]

    attempt_ids: list[str] = []
    errors: list[str] = []
    lock = threading.Lock()

    def worker(i: int) -> None:
        try:
            kwargs = _make_lesson_kwargs(i)
            lid = bus._lesson_id(
                kwargs["source_layer"], kwargs["title"], kwargs["body"]
            )
            # Bypass the outer dedup in write_lesson so all writers always call _append_lesson
            lesson = dict(
                ts=int(time.time()),
                lesson_id=lid,
                **{k: kwargs[k] for k in
                   ("source_layer", "target_layers", "category", "title",
                    "body", "transfer_note", "evidence_path", "provenance_sha")},
            )
            bus._append_lesson(lesson)
            with lock:
                attempt_ids.append(lid)
        except Exception as exc:
            with lock:
                errors.append(f"worker {i}: {exc}")

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    landed = _count_landed(tmp_jsonl)

    # Restore
    bus._append_lesson = original_append  # type: ignore[method-assign]
    bus.JSONL_PATH = original_path
    bus._LOCK_PATH = original_lock

    if errors:
        print(f"  Worker errors: {errors}", file=sys.stderr)

    return len(attempt_ids), len(landed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    overall_pass = True

    # ---- 1. Demonstrate the RACE (unfixed) --------------------------------
    with tempfile.TemporaryDirectory() as td:
        tmp_unfixed = Path(td) / "cross_pollination_unfixed.jsonl"
        attempts_unfixed, landed_unfixed = _run_unfixed(tmp_unfixed)
    lost_unfixed = attempts_unfixed - landed_unfixed
    print(f"[RACE DEMO] N={N} concurrent writers (unfixed, racy):")
    print(f"  attempts={attempts_unfixed}  landed={landed_unfixed}  lost={lost_unfixed}")
    if lost_unfixed > 0:
        print(f"  -> RACE CONFIRMED: {lost_unfixed} lesson(s) were silently lost (as expected without lock)")
    else:
        print(f"  -> Race not triggered this run (timing-dependent; the fix is still needed)")
    print()

    # ---- 2. Verify the FIX over MULTIPLE rounds (the strict gate) ----------
    # A single round passing is NOT proof (the race was intermittent). Require every
    # one of ROUNDS back-to-back 10-way rounds to land 100%, plus a 20-way round.
    print(f"[FIX VERIFICATION] {ROUNDS} rounds of N={N} concurrent writers (must ALL land every round):")
    total_expected = 0
    total_landed = 0
    for r in range(1, ROUNDS + 1):
        with tempfile.TemporaryDirectory() as td:
            tmp_fixed = Path(td) / "cross_pollination_fixed.jsonl"
            n_returned, n_landed, errors = _run_fixed(tmp_fixed, N)
        total_expected += N
        total_landed += n_landed
        lost = N - n_landed
        status = "PASS" if (n_landed == N and not errors) else "FAIL"
        print(f"  round {r}/{ROUNDS}: returned={n_returned}  landed={n_landed}/{N}  lost={lost}  [{status}]")
        if errors:
            print(f"    worker errors: {errors}", file=sys.stderr)
        if n_landed != N or errors:
            overall_pass = False

    print(f"  AGGREGATE: {total_landed}/{total_expected} landed across {ROUNDS} rounds "
          f"({'all rounds 100%' if total_landed == total_expected else 'LOSS DETECTED'})")

    if total_landed == total_expected:
        print(f"  [PASS] every round landed all {N} -- no concurrent-write loss across {ROUNDS} rounds.")
    else:
        print(f"  [FAIL] {total_expected - total_landed} lesson(s) lost across {ROUNDS} rounds -- fix is broken.")
    print()

    # ---- 2b. High-contention single round: 20 concurrent writers ----------
    print(f"[FIX VERIFICATION -- HIGH CONTENTION] one round of N={N_BIG} concurrent writers:")
    with tempfile.TemporaryDirectory() as td:
        tmp_big = Path(td) / "cross_pollination_big.jsonl"
        n_returned_big, n_landed_big, errors_big = _run_fixed(tmp_big, N_BIG)
    lost_big = N_BIG - n_landed_big
    if n_landed_big == N_BIG and not errors_big:
        print(f"  returned={n_returned_big}  landed={n_landed_big}/{N_BIG}  lost={lost_big}  [PASS]")
    else:
        print(f"  returned={n_returned_big}  landed={n_landed_big}/{N_BIG}  lost={lost_big}  [FAIL]")
        if errors_big:
            print(f"    worker errors: {errors_big}", file=sys.stderr)
        overall_pass = False
    print()

    # ---- 3. Bus self-test (regression guard) ------------------------------
    print("[BUS SELFTEST]")
    fails = bus._selftest(verbose=True)
    if fails == 0:
        print("  [PASS] selftest: no regression")
    else:
        print(f"  [FAIL] selftest: {fails} failure(s)")
        overall_pass = False

    print()
    print(f"[RESULT] {'PASS' if overall_pass else 'FAIL'} "
          f"(strict gate: {ROUNDS}x{N} all-land + {N_BIG}-way all-land + selftest)")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
