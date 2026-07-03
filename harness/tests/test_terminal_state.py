"""FAIL-CLOSED TERMINAL STATE: the run status must NEVER label an empty plan, a parked-HITL frontier, a believed-
but-unverified completion, or a refuted vein as ground-truth 'solved'. Pins the 2026-07-03 fail-closed classifier
(the audit found all three false victories under the old `all_work_done = (not open and not refuted)` rule)."""
from __future__ import annotations

from metaop.graph import (STATUS_BLOCKED, STATUS_EMPTY_PLAN, STATUS_FAILED, STATUS_SOLVED,
                          STATUS_SOLVED_UNVERIFIED, _terminal_run_status)


def _done(verdict="pass", evidence="mechanical", **extra):
    n = {"id": "x", "status": "done", "verdict": verdict, "evidence_type": evidence}
    n.update(extra)
    return n


def test_empty_plan_is_not_solved():
    assert _terminal_run_status([]) == STATUS_EMPTY_PLAN
    assert _terminal_run_status([None, "junk"]) == STATUS_EMPTY_PLAN


def test_parked_hitl_blocks_solved():
    fr = [_done(), {"id": "p", "status": "awaiting_approval"}]
    assert _terminal_run_status(fr) == STATUS_BLOCKED


def test_refuted_is_failed_not_solved():
    assert _terminal_run_status([{"id": "r", "status": "refuted"}]) == STATUS_FAILED


def test_unverified_pass_is_not_solved():
    fr = [_done(verdict="inconclusive", evidence="llm_panel_unverified", unverified_pass=True)]
    assert _terminal_run_status(fr) == STATUS_SOLVED_UNVERIFIED


def test_all_verified_is_solved():
    assert _terminal_run_status([_done(), _done()]) == STATUS_SOLVED


def test_any_unverified_dominates_verified():
    fr = [_done(), _done(verdict="inconclusive", unverified_pass=True)]
    assert _terminal_run_status(fr) == STATUS_SOLVED_UNVERIFIED


def test_blocked_dominates_unverified():
    fr = [_done(verdict="inconclusive", unverified_pass=True), {"id": "p", "status": "awaiting_approval"}]
    assert _terminal_run_status(fr) == STATUS_BLOCKED
