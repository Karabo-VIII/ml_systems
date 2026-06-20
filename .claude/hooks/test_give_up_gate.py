#!/usr/bin/env python3
"""Unit + wiring tests for the GIVE-UP / SCOPE-COLLAPSE gate in autonomy_loop.py.

Run: python .claude/hooks/test_give_up_gate.py   (exit 0 = all pass).
Tests the pure decision matrix, the P1 turn-group text accumulation, artifact coverage, the DONE-override,
and the block wiring (in-process, monkeypatched coverage + temp state). No emoji (Windows cp1252).
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HOOKS = Path(__file__).resolve().parent
ROOT = HOOKS.parents[1]
sys.path.insert(0, str(HOOKS))
import autonomy_loop as al  # noqa: E402

_OK = [True]


def check(name, cond):
    _OK[0] = _OK[0] and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def _jsonl(records):
    fd, p = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return p


def _asst(text=None, thinking=None, sidechain=False):
    content = []
    if thinking is not None:
        content.append({"type": "thinking", "thinking": thinking})
    if text is not None:
        content.append({"type": "text", "text": text})
    return {"type": "assistant", "isSidechain": sidechain,
            "message": {"stop_reason": "end_turn", "content": content}}


def _user(text):
    return {"type": "user", "message": {"content": text}}


def test_read_last_text():
    # P1: a thinking-only end_turn record precedes the text record in the SAME turn-group -> accumulate the text.
    p = _jsonl([_user("go"), _asst(thinking="(reasoning, no text)"),
                _asst(text="no tradeable setup chaser across u100; BTC is representative.")])
    got = al._gg_read_last_text(p)
    os.unlink(p)
    check("P1 accumulates text past a thinking-only end_turn record",
          "no tradeable setup chaser" in got)

    # sidechain (subagent) records are skipped; only the main-thread close text is returned.
    p = _jsonl([_user("go"), _asst(text="sidechain noise", sidechain=True), _asst(text="main close text")])
    got = al._gg_read_last_text(p)
    os.unlink(p)
    check("sidechain records skipped", got.strip() == "main close text")

    # the walk stops at the preceding user record (does not bleed into an earlier turn).
    p = _jsonl([_asst(text="OLD earlier-turn close"), _user("new prompt"), _asst(text="NEW close")])
    got = al._gg_read_last_text(p)
    os.unlink(p)
    check("stops at preceding user record", got.strip() == "NEW close" and "OLD" not in got)

    check("missing transcript -> '' (fail-open)", al._gg_read_last_text("/no/such/file") == "")


def test_decision_matrix():
    # F1 collapse on a breadth goal with NARROW work (1 asset) -> BLOCK
    b, s = al._gg_decision("no tradeable setup chaser across u100; BTC is representative.", True, 1)
    check("F1 breadth+collapse+1-asset -> BLOCK", b and s)

    # F2 give-up on breadth but FULL surface attempted (>=floor) -> ALLOW (earned null)
    b, _ = al._gg_decision("swept all of u100, honest null; de-risked beta confirmed across the surface", True, 25)
    check("F2 breadth+giveup+full-surface(25) -> ALLOW (earned null)", not b)

    # F3 give-up words but ZERO strategy work (a planning/discussion turn) -> ALLOW (exempt)
    b, _ = al._gg_decision("this looks impossible to me", True, 0)
    check("F3 breadth+giveup+0-assets (discussion) -> ALLOW (exempt)", not b)

    # F4 constraint-relaxation shortcut -> BLOCK regardless of coverage / breadth
    b, _ = al._gg_decision("the only unlock is to enable the short side", True, 0)
    check("F4 shortcut(breadth) -> BLOCK", b)
    b, _ = al._gg_decision("the only unlock is to enable the short side", False, 0)
    check("F4 shortcut(non-breadth) -> BLOCK", b)

    # F5 clean factual lookup answer (T6 false-positive control) -> ALLOW
    b, _ = al._gg_decision("u100.yaml currently has 70 assets with data.", True, 0)
    check("F5 clean lookup answer (T6 control) -> ALLOW", not b)

    # extra: collapse phrase but NON-breadth goal -> no collapse signal -> ALLOW
    b, _ = al._gg_decision("BTC is representative here", False, 1)
    check("collapse phrase on non-breadth goal -> ALLOW", not b)


def test_coverage():
    syms = {f"AS{i:02d}" for i in range(25)} | {"BTC"}
    with tempfile.TemporaryDirectory() as d:
        # a fresh strategy-output file touching 25 distinct non-BTC symbols -> >= floor
        Path(d, "sweep.json").write_text(json.dumps({"symbols": sorted(syms)}), encoding="utf-8")
        import time as _t
        n = al._gg_coverage(_t.time(), symbols=syms, runs_root=d)
        check("coverage counts >=floor on a 25-symbol sweep", n >= al.ASSET_FLOOR)

    # 2026-06-20 regression: artifacts write SUFFIXED tickers (BCHUSDT); a bare \b match undercounts -> false block
    with tempfile.TemporaryDirectory() as d:
        bases = {"BX%02d" % i for i in range(25)}
        Path(d, "engine.json").write_text(json.dumps({"admitted_symbols": [b + "USDT" for b in bases]}), encoding="utf-8")
        import time as _t
        n = al._gg_coverage(_t.time(), symbols=bases | {"BTC"}, runs_root=d)
        check("coverage counts USDT-suffixed tickers (BXxxUSDT)", n >= al.ASSET_FLOOR)

    with tempfile.TemporaryDirectory() as d:
        Path(d, "btc_only.csv").write_text("BTC,BTC,BTC\n", encoding="utf-8")
        import time as _t
        n = al._gg_coverage(_t.time(), symbols=syms, runs_root=d)
        check("coverage = 0 when only BTC touched", n == 0)

    with tempfile.TemporaryDirectory() as d:
        sub = Path(d, "autonomy")
        sub.mkdir()
        (sub / "trace.jsonl").write_text(json.dumps({"symbols": sorted(syms)}), encoding="utf-8")
        import time as _t
        n = al._gg_coverage(_t.time(), symbols=syms, runs_root=d)
        check("coverage excludes runs/autonomy/ (transcripts don't count)", n == 0)


def test_done_override():
    check("bare 'VERIFIED' does NOT override", al._gg_done_override("VERIFIED -- all good") is False)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True,
                          creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
    sha = (head.stdout or "").strip()
    if len(sha) >= 7:
        check("DONE + a real git SHA overrides", al._gg_done_override(f"SOLVED -- committed {sha[:12]}") is True)
    else:
        check("DONE + git SHA (skipped: no git)", True)


def test_block_wiring():
    # In-process end-to-end: a collapse close on a breadth goal with narrow (monkeypatched) coverage must BLOCK.
    saved = (al._gg_coverage, al.CURRENT_MD, al.CHARTER_MD, al.GIVEUP_STATE)
    try:
        with tempfile.TemporaryDirectory() as d:
            cur = Path(d, "CURRENT.md"); cur.write_text("CURRENT GOAL: trade any slice across u100", encoding="utf-8")
            al._gg_coverage = lambda *a, **k: 1           # narrow work: 1 asset
            al.CURRENT_MD = str(cur)
            al.CHARTER_MD = str(Path(d, "CHARTER.md"))     # absent -> ''
            al.GIVEUP_STATE = str(Path(d, "giveup.json"))  # fresh counter
            p = _jsonl([_user("build the u100 slice trader"),
                        _asst(text="No tradeable setup chaser across u100; BTC is representative, so just trade BTC.")])
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    al.give_up_gate({"transcript_path": p})
            except SystemExit:
                pass
            os.unlink(p)
            out = buf.getvalue()
            check("collapse close BLOCKS (decision=block emitted)", '"decision": "block"' in out)
            check("block reason carries the GIVE-UP GATE banner", "GIVE-UP GATE" in out)

            # a clean factual close on the same goal must NOT block
            p = _jsonl([_user("how many assets?"), _asst(text="u100.yaml currently has 70 assets with data.")])
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    al.give_up_gate({"transcript_path": p})
            except SystemExit:
                pass
            os.unlink(p)
            check("clean lookup close does NOT block (T6 control)", '"decision": "block"' not in buf.getvalue())
    finally:
        al._gg_coverage, al.CURRENT_MD, al.CHARTER_MD, al.GIVEUP_STATE = saved


def test_subprocess_failopen():
    # the live Stop hook must never crash / never trap: stop_hook_active -> clean exit 0, no block.
    payload = {"stop_hook_active": True}
    r = subprocess.run([sys.executable, str(HOOKS / "autonomy_loop.py")], input=json.dumps(payload),
                       capture_output=True, text=True, timeout=30,
                       creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
    check("stop_hook_active -> exit 0, no block", r.returncode == 0 and '"decision": "block"' not in r.stdout)


def main():
    print("== give_up_gate tests ==")
    test_read_last_text()
    test_decision_matrix()
    test_coverage()
    test_done_override()
    test_block_wiring()
    test_subprocess_failopen()
    print("ALL PASS" if _OK[0] else "*** SOME FAILED ***")
    return 0 if _OK[0] else 1


if __name__ == "__main__":
    raise SystemExit(main())
