#!/usr/bin/env python3
"""Resume-all + loop-brief metadata -- the canonical RECOVERY + CELL-STYLE LINEAGE layer for the autonomous stack.

Two jobs:
1) RESUMABILITY (subscription-limit / crash recovery): every metaop loop checkpoints durably to
   runs/autonomy/metaop_<thread>.db. This lists every parked loop and resumes it from its saved frontier
   (state survives a dead orchestrator -- the loops are independent processes; the ledger/logs/lanes are on disk).
2) LOOP BRIEFS (Cell-style evolution): each loop instance gets runs/autonomy/loop_briefs/<thread>.json with its
   objective + parent generation + a digest of the prior generation's lessons (from the learnings lane). A reborn
   loop digests its predecessor end-to-end -- loops are points in time that compound.

Usage:
  python scripts/autonomy/resume_all.py --list                 # show parked loops + briefs
  python scripts/autonomy/resume_all.py --resume --budget 20   # resume every parked loop
  python scripts/autonomy/resume_all.py --brief T --objective "..." [--parent P]   # write a loop brief
No emoji (cp1252).
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "autonomy"))
BRIEFS = os.path.join(ROOT, "runs", "autonomy", "loop_briefs")
RUNNER = os.path.join(ROOT, "scripts", "autonomy", "run_metaop.py")


def _threads():
    out = []
    for db in glob.glob(os.path.join(ROOT, "runs", "autonomy", "metaop_*.db")):
        t = os.path.basename(db)[len("metaop_"):-len(".db")]
        out.append(t)
    return sorted(out)


def write_brief(thread, objective, parent=None, channel=None):
    """Cell-style: attach a brief + digest the parent generation's lessons from the learnings lane."""
    os.makedirs(BRIEFS, exist_ok=True)
    digest = []
    try:
        from metaop import learnings
        ch = channel or thread.split("-")[-1]  # e.g. ama2-expert -> expert
        digest = [r.get("lesson", "")[:200] for r in learnings.recent(8, ch)]
    except Exception:
        pass
    brief = {"thread": thread, "objective": objective, "parent": parent, "generation_ts": int(time.time()),
             "prior_lessons_digest": digest}
    with open(os.path.join(BRIEFS, f"{thread}.json"), "w", encoding="utf-8") as fh:
        json.dump(brief, fh, indent=2)
    return brief


def list_loops():
    threads = _threads()
    print(f"=== parked/active metaop loops ({len(threads)}) ===")
    for t in threads:
        bf = os.path.join(BRIEFS, f"{t}.json")
        b = json.load(open(bf)) if os.path.exists(bf) else {}
        print(f"  {t}: brief={'yes' if b else 'no'} | obj={str(b.get('objective',''))[:60]} | "
              f"prior_lessons={len(b.get('prior_lessons_digest',[]))}")
    print("\nRESUME PROCEDURE: loops are durable + independent. If the orchestrator dies, the loops keep running; "
          "resume monitoring by reading docs/SELF_EVOLUTION_LEDGER.md + experiments/*/OVERSEER_LOG.md. Resume a "
          "dead loop with: python scripts/autonomy/run_metaop.py resume --thread <T> --budget N")


def resume_all(budget):
    for t in _threads():
        print(f"--- resume {t} (budget {budget}) ---")
        subprocess.run([sys.executable, RUNNER, "resume", "--thread", t, "--budget", str(budget), "--backend", "cli"],
                       cwd=ROOT, creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--brief"); ap.add_argument("--objective", default=""); ap.add_argument("--parent")
    a = ap.parse_args()
    if a.brief:
        print(json.dumps(write_brief(a.brief, a.objective, a.parent), indent=2)[:400]); return 0
    if a.resume:
        resume_all(a.budget); return 0
    list_loops(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
