"""
Long-Running Job Registry CLI.

Pick the next PENDING job, mark in-progress, or update status. Each
instance should do:
  1. python scripts/long_running_jobs.py --list
  2. Read the top PENDING job, do its next_step
  3. python scripts/long_running_jobs.py --update <id> status COMPLETE
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY = PROJECT_ROOT / "scripts" / "long_running_jobs.yaml"


def load():
    with open(REGISTRY) as f:
        return yaml.safe_load(f)


def save(data):
    with open(REGISTRY, "w") as f:
        yaml.dump(data, f, sort_keys=False, width=100, default_flow_style=False)


def list_jobs(data):
    print(f"\n{'ID':<36} {'Pri':<4} {'Status':<12} {'Title'}")
    print("-" * 120)
    for j in data["jobs"]:
        jid = j["id"]
        pri = j["priority"]
        st = j["status"]
        title = j["title"][:70]
        print(f"{jid:<36} {pri:<4} {st:<12} {title}")


def pick_next(data):
    """Return the highest-priority PENDING job."""
    pri_rank = {"P0": 0, "P1": 1, "P2": 2}
    pending = [j for j in data["jobs"] if j["status"] == "PENDING"]
    if not pending:
        print("\nNo PENDING jobs. All in IN_PROGRESS / BLOCKED / COMPLETE.")
        return None
    pending.sort(key=lambda j: pri_rank.get(j["priority"], 9))
    top = pending[0]
    print(f"\n[PICK] {top['id']}  ({top['priority']})  {top['title']}")
    print(f"  Purpose:  {top['purpose'].strip().splitlines()[0][:90]}")
    print(f"  ETA:      {top.get('eta_sessions', '?')} sessions, "
          f"~{top.get('estimated_wall_clock_min', '?')} min wall-clock")
    print(f"\n  Next step:")
    for line in top["next_step"].strip().splitlines():
        print(f"    {line}")
    print(f"\n  Artifacts:")
    for a in top.get("artifacts", []):
        print(f"    {a}")
    return top


def update(data, job_id, field, value):
    for j in data["jobs"]:
        if j["id"] == job_id:
            j[field] = value
            print(f"Updated {job_id}: {field} = {value}")
            return True
    print(f"Job id {job_id!r} not found.")
    return False


def main():
    ap = argparse.ArgumentParser(description="Long-running job registry CLI.")
    ap.add_argument("--list", action="store_true", help="List all jobs")
    ap.add_argument("--pick", action="store_true",
                     help="Pick next PENDING job")
    ap.add_argument("--update", nargs=3, metavar=("ID", "FIELD", "VALUE"),
                     help="Update a job field")
    args = ap.parse_args()

    data = load()
    if args.list:
        list_jobs(data)
        return
    if args.pick:
        pick_next(data)
        return
    if args.update:
        jid, field, val = args.update
        if update(data, jid, field, val):
            save(data)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
