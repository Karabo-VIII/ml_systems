#!/usr/bin/env python3
"""Cross-session autonomy DRIVER -- the durable engine (works while you sleep).

The Stop hook keeps a SINGLE live session going. This driver is the CROSS-SESSION version: it spawns fresh
headless Claude sessions (`claude -p`) in a loop, one per frontier node, so the loop survives context limits,
crashes, and session ends -- Lilian Weng's "continuous loop with state living in the repo". Each cycle:
read frontier -> pick top-EV open node -> run a fresh `claude -p` cycle on it -> the cycle updates the frontier
-> repeat until a real stop-condition.

USAGE:  python scripts/autonomy/autonomy_driver.py [--frontier runs/autonomy/frontier.json] [--max N]
Stops when: frontier empty/below floor, budget spent, max cycles, or the model marks the objective SOLVED.
Safety: hard --max ceiling, per-cycle timeout, and it NEVER auto-merges trust-critical -- the cycle prompt
enforces the sandbox->review->push gate. No emoji (Windows cp1252).

Prereq: the `claude` CLI on PATH (headless mode). For tighter control use the Claude Agent SDK `query()` instead
of subprocess -- swap run_cycle() accordingly.
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HARD_CEIL = 500


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def pick_node(f):
    floor = float(f.get("value_floor", 0.0))
    op = [n for n in f.get("nodes", []) if n.get("status") == "open" and float(n.get("ev", 0)) >= floor]
    return max(op, key=lambda n: float(n.get("ev", 0))) if op else None


def stop_reason(f, cycles, hardmax):
    b = f.get("budget", {}) or {}
    if int(b.get("spent", 0)) >= int(b.get("max_cycles", 0) or HARD_CEIL):
        return "budget spent"
    if cycles >= hardmax:
        return f"hard cycle ceiling {hardmax}"
    if str(f.get("status", "")).lower() == "solved":
        return "objective marked SOLVED"
    if pick_node(f) is None:
        return "frontier empty / below value_floor"
    return None


def cycle_prompt(node, f):
    ov = f.get("overseer", {}) or {}
    acceptance = ov.get("acceptance_test", f.get("success_criteria", "(acceptance_test unset)"))
    return (
        f"[AUTONOMY DRIVER CYCLE -- you are the OVERSEER (Tier-0, .claude/skills/_common/OVERSEER.md), fresh session]\n"
        f"OBJECTIVE: {f.get('objective')}\nSUCCESS CRITERIA: {f.get('success_criteria')}\n"
        f"ACCEPTANCE TEST (DONE only when this is VERIFIED, not asserted): {acceptance}\n"
        f"THIS NODE (EV={node.get('ev')}, id={node.get('id')}, kind={node.get('kind')}): {node.get('task')}\n"
        "DO: (1) re-state the objective + confirm this node serves it (drift guard); (2) DISPATCH execution to a "
        "worker (Agent/Workflow) -- you OVERSEE, you do not do primary building/running yourself (tiny lookups OK); "
        "(3) JUDGE the return against the ACCEPTANCE TEST adversarially with RWYB (verify against artifacts; refuse "
        "false victory / drift-to-proxy / single-path narrowness); (4) UPDATE runs/autonomy/frontier.json -- mark "
        "this node done|refuted|blocked WITH evidence, append overseer.fulfillment_ledger (verdict + git SHA/run "
        "output + a real date), PUSH new neighbor nodes (a -k falsifier + a +k generalization), increment "
        "budget.spent, re-rank; (5) WRITE-FORWARD learnings to memory/. If the acceptance_test is genuinely VERIFIED "
        "set frontier.status = 'solved'. You MAKE THE CALLS (user's proxy): COMMIT changes -- git is the revert "
        "safety-net -- and escalate to the real user ONLY for a genuinely irreversible real-world action (deploy "
        "real capital / external send / shared-history rewrite); never block on a decision git can revert."
    )


def run_cycle(prompt, timeout):
    try:
        subprocess.run(["claude", "-p", prompt], cwd=ROOT, timeout=timeout,
                       creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
        return True
    except FileNotFoundError:
        print("ERROR: `claude` CLI not on PATH. Install Claude Code or swap run_cycle() for the Agent SDK query().")
        return False
    except subprocess.TimeoutExpired:
        print("  cycle timed out; continuing to next")
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontier", default=os.path.join(ROOT, "runs", "autonomy", "frontier.json"))
    ap.add_argument("--max", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--dry-run", action="store_true", help="print the cycle plan without spawning claude")
    args = ap.parse_args()

    cycles = 0
    while True:
        try:
            f = load(args.frontier)
        except Exception as e:
            print(f"STOP: cannot read frontier ({e})")
            break
        sr = stop_reason(f, cycles, args.max)
        if sr:
            print(f"STOP: {sr}  (after {cycles} cycles)")
            break
        node = pick_node(f)
        print(f"--- cycle {cycles+1}: node {node.get('id')} (EV={node.get('ev')}): {node.get('task')[:80]}")
        if args.dry_run:
            print(cycle_prompt(node, f)[:400] + " ...")
            break
        if not run_cycle(cycle_prompt(node, f), args.timeout):
            break
        cycles += 1
        time.sleep(2)


if __name__ == "__main__":
    main()
