#!/usr/bin/env python3
"""Regression test for the PID-reuse stuck-path fix (G-J 2026-06-07).

Proves: a tracked job whose PID got RECYCLED to a different process is reported DEAD (not falsely ALIVE -> no
eternal WAIT-MODE), while a genuinely-live tracked job is ALIVE, and legacy locks without a `created` field still
work (bare check). Drives the REAL track_job lock dir in a temp HOME so the live state is untouched. No emoji.
"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import proc_liveness as pl
import track_job as tj


def main():
    fails = []

    # 1. proc_liveness core: self is alive; self with WRONG create-time is dead (reuse); bogus pid dead.
    me = os.getpid()
    ct = pl.create_time(me)
    if not pl.alive(me):
        fails.append("alive(self) should be True")
    if ct is not None and not pl.alive(me, ct):
        fails.append("alive(self, correct_ct) should be True")
    if ct is not None and pl.alive(me, ct + 99999):
        fails.append("alive(self, WRONG_ct) should be False (reuse must be detected)")
    if pl.alive(999999):
        fails.append("alive(bogus pid) should be False")

    # 2. track_job end-to-end against a temp locks dir.
    tmp = tempfile.mkdtemp(prefix="gjlock_")
    tj.LOCKS = tmp  # redirect lock dir so we don't touch live state
    tj.track("gj_live", pid=me, cmd="self")
    ids = [j["id"] for j in tj.alive_jobs()]
    if "gj_live" not in ids:
        fails.append(f"a genuinely-live tracked job should be listed alive; got {ids}")

    # 3. simulate PID REUSE: rewrite the lock's created to a wrong value -> must be reaped as dead.
    lf = os.path.join(tmp, "gj_live.lock")
    d = json.loads(open(lf, encoding="utf-8").read())
    d["created"] = (d.get("created") or pl.create_time(me) or 0) + 99999  # pretend this PID is now a DIFFERENT proc
    open(lf, "w", encoding="utf-8").write(json.dumps(d))
    ids2 = [j["id"] for j in tj.alive_jobs()]
    if "gj_live" in ids2:
        fails.append("a RECYCLED-PID lock must NOT be reported alive (this is the eternal-WAIT-MODE stuck-path)")
    if os.path.exists(lf):
        fails.append("a dead-by-reuse lock should be reaped by alive_jobs()")

    # 4. legacy lock with NO `created` field still works (bare check -> alive while pid lives).
    tj.track("gj_legacy", pid=me, cmd="legacy")
    lf2 = os.path.join(tmp, "gj_legacy.lock")
    d2 = json.loads(open(lf2, encoding="utf-8").read())
    d2.pop("created", None)
    open(lf2, "w", encoding="utf-8").write(json.dumps(d2))
    if "gj_legacy" not in [j["id"] for j in tj.alive_jobs()]:
        fails.append("a legacy lock (no created field) must still be honored via the bare check")

    if fails:
        print(f"[proc_liveness/G-J] FAIL ({len(fails)}):")
        for f in fails:
            print("   -", f)
        return 1
    print("[proc_liveness/G-J] ALL PASS (reuse detected, live honored, dead reaped, legacy compatible)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
