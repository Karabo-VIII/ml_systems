"""FALSIFIER probe for PersistentCliBrain (#14). Deterministic -- no real `claude` spawned.

We monkeypatch:
  - brain.find_claude  -> a fake exe path (so construction succeeds without the real CLI)
  - brain.subprocess.run -> a programmable fake that returns whatever envelope/stdout/exception we script
  - CliBrain._run (the SUPER fallback) -> a sentinel that records it was hit and returns a marker string
so every assertion is about the REAL PersistentCliBrain._run control flow, observed by running it.
No emoji (cp1252).
"""
from __future__ import annotations
import json
import sys
import types
import subprocess as _subprocess_real

import importlib
# G-A dedup (2026-06-07): PersistentCliBrain/CliBrain + the subprocess/find_claude module globals the probe
# monkeypatches now live in the CANONICAL engine harness.metaop.brain (the scripts.autonomy.metaop.brain shim only
# re-exports the classes). Patch the canonical module so we exercise the REAL PersistentCliBrain._run control flow.
brain = importlib.import_module("harness.metaop.brain")
from scripts.autonomy.metaop.brain import PersistentCliBrain, CliBrain  # same class objects (re-exported)

FAKE_EXE = r"C:\fake\claude.exe"

# ---- sentinel for the plain-text CliBrain fallback path -------------------------------------------
FALLBACK_MARKER = "<<<CLIBRAIN_PLAINTEXT_FALLBACK>>>"
fallback_hits = {"n": 0, "prompts": []}
def fake_super_run(self, prompt, retries=1):
    fallback_hits["n"] += 1
    fallback_hits["prompts"].append(prompt[:40])
    return FALLBACK_MARKER
CliBrain._run = fake_super_run  # the persistent class calls super()._run -> this sentinel

# ---- programmable fake subprocess.run -------------------------------------------------------------
class FakeCompleted:
    def __init__(self, stdout): self.stdout = stdout; self.stderr = ""; self.returncode = 0

_script = {"mode": None, "captured_cmds": []}
def fake_run(cmd, capture_output=True, text=True, timeout=None, env=None, **kw):
    _script["captured_cmds"].append(list(cmd))
    mode = _script["mode"]
    if mode == "good":
        # a healthy fresh/resume envelope: returns a session_id + result text
        sid = "sess-RESUMED" if "--resume" in cmd else "sess-FRESH-001"
        return FakeCompleted(json.dumps({"type":"result","subtype":"success","is_error":False,
                                         "session_id": sid, "result": f"OK from {sid}"}))
    if mode == "malformed":
        return FakeCompleted("this is { not valid json at all >>>")
    if mode == "empty":
        return FakeCompleted("")
    if mode == "missing_sid":
        return FakeCompleted(json.dumps({"type":"result","subtype":"success","is_error":False,
                                         "result": "answer but NO session_id field"}))
    if mode == "error_env":
        return FakeCompleted(json.dumps({"type":"result","subtype":"error_during_execution",
                                         "is_error":True,"session_id":"sess-DEAD","result":""}))
    if mode == "ctx_limit":
        return FakeCompleted(json.dumps({"type":"result","subtype":"success",
                                         "is_error":True,"api_error_status":429,"result":"context window exceeded"}))
    if mode == "empty_result_text":
        return FakeCompleted(json.dumps({"type":"result","subtype":"success","is_error":False,
                                         "session_id":"sess-X","result":"   "}))
    if mode == "timeout":
        raise _subprocess_real.TimeoutExpired(cmd, timeout)
    if mode == "boom":
        raise OSError("simulated subprocess failure (exe vanished)")
    raise AssertionError(f"unhandled mode {mode}")

brain.subprocess.run = fake_run
brain.find_claude = lambda: FAKE_EXE

def reset(mode):
    _script["mode"] = mode
    _script["captured_cmds"].clear()
    fallback_hits["n"] = 0
    fallback_hits["prompts"].clear()

def new_brain():
    return PersistentCliBrain(exe=FAKE_EXE)

results = []
def check(name, cond, detail=""):
    results.append((name, cond, detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")

print("="*80)
print("(b)-mechanism: fresh node-1 omits --resume; node-2 carries --resume <captured sid>")
print("="*80)
reset("good")
b = new_brain()
out1 = b._run("PROMPT NODE 1")
cmd1 = _script["captured_cmds"][-1]
check("node-1 is FRESH (no --resume)", "--resume" not in cmd1, f"cmd1={cmd1[:3]}... resume={'--resume' in cmd1}")
check("node-1 stored session_id", b.session_id == "sess-FRESH-001", f"session_id={b.session_id}")
check("node-1 returned inner result TEXT (not envelope)", out1 == "OK from sess-FRESH-001", f"out1={out1!r}")
out2 = b._run("PROMPT NODE 2")
cmd2 = _script["captured_cmds"][-1]
check("node-2 RESUMES with the carried sid", "--resume" in cmd2 and "sess-FRESH-001" in cmd2,
      f"resume_arg={[c for c in cmd2 if 'sess' in str(c)]}")
check("node-2 updated session_id to resumed sid", b.session_id == "sess-RESUMED", f"session_id={b.session_id}")
check("no fallback used on healthy path", fallback_hits["n"] == 0, f"fallback_hits={fallback_hits['n']}")

print("\n" + "="*80)
print("(a) FALLBACK survival on backend failures -- loop must NOT crash or hang")
print("="*80)
for mode, label in [("malformed","malformed JSON stdout"),
                    ("empty","empty stdout"),
                    ("missing_sid","valid JSON but NO session_id (still must return text? or fallback)"),
                    ("error_env","is_error envelope"),
                    ("ctx_limit","api_error_status (context-limit)"),
                    ("empty_result_text","success env but blank result text"),
                    ("timeout","subprocess TimeoutExpired"),
                    ("boom","subprocess OSError")]:
    reset(mode)
    b = new_brain()
    try:
        out = b._run(f"PROMPT [{mode}]")
        crashed = False; err = ""
    except Exception as e:
        out = None; crashed = True; err = repr(e)
    survived = (not crashed)
    print(f"  mode={mode:18s} -> survived={survived}  out={str(out)[:50]!r}  fallback_hits={fallback_hits['n']}  err={err}")
    check(f"(a) survives '{label}'", survived and not crashed, f"out={str(out)[:30]!r}")

print("\n" + "="*80)
print("(a/c) missing_sid SPECIAL: does a missing session_id silently break continuity?")
print("="*80)
reset("missing_sid")
b = new_brain()
out = b._run("first call, env has result but no session_id")
check("missing_sid: returns the result text (no crash)", out == "answer but NO session_id field" or out == FALLBACK_MARKER,
      f"out={out!r}")
check("missing_sid: session_id stays None (next call is fresh, not a stale resume)",
      b.session_id is None, f"session_id={b.session_id}")

print("\n" + "="*80)
print("(c) REBIRTH fires on context-limit / error -- and ONLY then")
print("="*80)
# prime a live session first
reset("good"); b = new_brain(); b._run("prime"); b._run("prime2")
assert b.session_id == "sess-RESUMED", b.session_id
before_rebirths = b.rebirths
# now feed a context-limit envelope
_script["mode"] = "ctx_limit"; fallback_hits["n"]=0
out = b._run("this one overflows context")
check("(c) ctx-limit -> rebirth incremented", b.rebirths == before_rebirths + 1, f"rebirths {before_rebirths}->{b.rebirths}")
check("(c) ctx-limit -> session dropped to None (next call cold/fresh)", b.session_id is None, f"sid={b.session_id}")
check("(c) ctx-limit -> served THIS node via plain-text fallback", fallback_hits["n"] == 1, f"fallback_hits={fallback_hits['n']}")
# error envelope path
reset("good"); b=new_brain(); b._run("p"); b._run("p2"); br0=b.rebirths
_script["mode"]="error_env"; fallback_hits["n"]=0
b._run("turn fails")
check("(c) error subtype -> rebirth + fallback", b.rebirths==br0+1 and fallback_hits["n"]==1, f"rebirths={b.rebirths} fb={fallback_hits['n']}")
# _is_error_envelope unit checks
check("(c) _is_error_envelope: success+no-error -> False",
      PersistentCliBrain._is_error_envelope({"subtype":"success","is_error":False}) is False)
check("(c) _is_error_envelope: is_error True -> True",
      PersistentCliBrain._is_error_envelope({"subtype":"success","is_error":True}) is True)
check("(c) _is_error_envelope: api_error_status -> True",
      PersistentCliBrain._is_error_envelope({"subtype":"success","api_error_status":529}) is True)
check("(c) _is_error_envelope: non-success subtype -> True",
      PersistentCliBrain._is_error_envelope({"subtype":"error_max_turns"}) is True)
check("(c) _is_error_envelope: non-dict -> True", PersistentCliBrain._is_error_envelope("not a dict") is True)

print("\n" + "="*80)
print("(c-edge) does rebirth fire when there was NO live session yet? (node-1 error)")
print("="*80)
reset("error_env")
b = new_brain()  # session_id is None
out = b._run("node-1 itself errors")
check("(c-edge) node-1 error: no spurious rebirth count (was None)", b.rebirths == 0, f"rebirths={b.rebirths}")
check("(c-edge) node-1 error: served via fallback, no crash", out == FALLBACK_MARKER, f"out={out!r}")

print("\n" + "="*80)
print("(a-edge) timeout WHILE resuming: must rebirth so we don't re-resume a dead session forever")
print("="*80)
reset("good"); b=new_brain(); b._run("p"); b._run("p2")
assert b.session_id == "sess-RESUMED"
br0=b.rebirths
_script["mode"]="timeout"; fallback_hits["n"]=0
out = b._run("hang")
check("(a-edge) timeout-on-resume -> rebirth (used_resume path)", b.rebirths==br0+1, f"rebirths={b.rebirths}")
check("(a-edge) timeout-on-resume -> fallback served", fallback_hits["n"]==1 and out==FALLBACK_MARKER)

print("\n" + "="*80)
n_pass = sum(1 for _,c,_ in results if c)
n_fail = sum(1 for _,c,_ in results if not c)
print(f"TOTAL: {n_pass} PASS, {n_fail} FAIL")
print("="*80)
sys.exit(1 if n_fail else 0)
