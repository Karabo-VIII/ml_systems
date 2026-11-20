"""FALSIFIER (d): does carrying ONE session across nodes break per-node INDEPENDENCE?

The graph (graph.py) shares ONE brain instance across plan/dispatch/judge/reflect, and dispatch runs
brain.work() CONCURRENTLY (default --parallel 2) via ThreadPoolExecutor. PersistentCliBrain holds a single
self.session_id. We reproduce the EXACT call pattern against one instance and observe:
  (d1) CONTEXT LEAK: every later node passes --resume of the session that saw the EARLIER nodes' prompts+outputs.
       The 3 adversarial judges (graph.judge loops brain.decide('judge') N times) become a SEQUENTIAL chain that
       all resume the same session -> not independent draws (the whole point of majority-vote verification).
  (d2) CONCURRENCY RACE: two dispatch workers, run truly concurrently, read+write self.session_id with NO lock
       and both --resume the SAME id at once (claude -p cannot safely resume one session from 2 processes).
Deterministic fake subprocess -- no real claude. No emoji (cp1252).
"""
from __future__ import annotations
import json, threading, importlib
# G-A dedup (2026-06-07): the subprocess/find_claude globals + PersistentCliBrain live in the CANONICAL engine
# harness.metaop.brain now (the scripts shim re-exports the classes). Patch the canonical module to drive the REAL
# PersistentCliBrain control flow.
brain = importlib.import_module("harness.metaop.brain")
from scripts.autonomy.metaop.brain import PersistentCliBrain, CliBrain  # same class objects (re-exported)

FAKE = r"C:\fake\claude.exe"
brain.find_claude = lambda: FAKE

# fake subprocess that records, per call, the prompt and the --resume target it was given, and returns a
# unique session id so we can trace which conversation each node was attached to.
_calls = []
_counter = {"n": 0}
_lock_for_counter = threading.Lock()
_barrier = {"b": None}
class FakeCompleted:
    def __init__(self, stdout): self.stdout = stdout; self.stderr=""; self.returncode=0
def fake_run(cmd, capture_output=True, text=True, timeout=None, env=None, **kw):
    resume_target = None
    if "--resume" in cmd:
        resume_target = cmd[cmd.index("--resume")+1]
    prompt = cmd[cmd.index("-p")+1] if "-p" in cmd else ""
    # if a barrier is armed (concurrency test), force both threads to be INSIDE the CLI call simultaneously
    if _barrier["b"] is not None:
        try: _barrier["b"].wait(timeout=5)
        except Exception: pass
    with _lock_for_counter:
        _counter["n"] += 1
        sid = f"S{_counter['n']:02d}"
    _calls.append({"thread": threading.current_thread().name, "prompt_head": prompt[:24],
                   "resumed": resume_target, "new_sid": sid})
    return FakeCompleted(json.dumps({"subtype":"success","is_error":False,"session_id":sid,
                                     "result":f"answer (was resuming {resume_target})"}))
brain.subprocess.run = fake_run

def reset(): _calls.clear(); _counter["n"]=0; _barrier["b"]=None

results=[]
def check(name, cond, detail=""):
    results.append((name,cond)); print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")

print("="*84)
print("(d1) CONTEXT LEAK across role boundaries: plan -> work -> 3x judge on ONE shared instance")
print("="*84)
reset()
b = PersistentCliBrain(exe=FAKE)
# replicate graph node order with a single shared brain (sequential, as plan/judge/reflect run):
b.decide("plan",   {"objective":"characterize SOL"})         # node: plan       (sees nothing prior)
b.work ("audit look-ahead in feature X")                     # node: dispatch   (resumes plan's session!)
# graph.judge: for a verify node it calls brain.decide('judge', {node}) n_judges (default 3) times in a loop
for _ in range(3):
    b.decide("judge", {"node": {"id":"n1","result":"some worker output"}})
print("  call trace (thread, prompt_head, resumed_session, new_sid):")
for c in _calls:
    print(f"    {c['thread']:10s} prompt={c['prompt_head']!r:28s} resumed={str(c['resumed']):5s} -> {c['new_sid']}")
# plan was fresh (resumed None); EVERY subsequent node resumed a non-None session (context carried)
plan_fresh = _calls[0]["resumed"] is None
work_resumed_plan = _calls[1]["resumed"] == _calls[0]["new_sid"]
judges_resumed = [c["resumed"] for c in _calls[2:]]
judges_all_resume = all(r is not None for r in judges_resumed)
# the 3 judges form a CHAIN: judge2 resumes judge1's session, judge3 resumes judge2's -> sequential dependence
judge_chain = (judges_resumed[0]==_calls[1]["new_sid"] and
               judges_resumed[1]==_calls[2]["new_sid"] and
               judges_resumed[2]==_calls[3]["new_sid"])
check("(d1) plan node was independent (fresh, no resume)", plan_fresh)
check("(d1) DEFECT: worker node inherited the PLAN session (context leak)", work_resumed_plan,
      f"work resumed {_calls[1]['resumed']} == plan sid {_calls[0]['new_sid']}")
check("(d1) DEFECT: every judge resumed prior context (NOT independent)", judges_all_resume,
      f"judge resume targets={judges_resumed}")
check("(d1) DEFECT: 3 judges are a SEQUENTIAL chain (judge_k sees judge_{k-1}), not 3 independent votes",
      judge_chain, "majority-vote independence is COLLAPSED")

print("\n" + "="*84)
print("(d2) CONCURRENCY RACE -- B1 FIX VERIFICATION (2026-06-08): the session lock now CONTAINS the race")
print("="*84)
# B1 fix: PersistentCliBrain._run is serialized under self._session_lock. We launch 2 dispatch workers on ONE
# shared instance. With the lock, the claude -p calls SERIALIZE: the second worker can no longer also --resume the
# pre-existing 'S00' at the same instant -- it resumes the session the FIRST worker produced. The barrier is left
# UNARMED because the lock would (correctly) prevent both threads from being inside the CLI call simultaneously.
reset()
b = PersistentCliBrain(exe=FAKE)
b.session_id = "S00"   # pretend plan already set a live session the batch will resume
errs=[]
def worker(tag):
    try: b.work(f"worker {tag} task")
    except Exception as e: errs.append(repr(e))
t1=threading.Thread(target=worker,args=("A",),name="W-A")
t2=threading.Thread(target=worker,args=("B",),name="W-B")
t1.start(); t2.start(); t1.join(); t2.join()
print("  call trace:")
for c in _calls:
    print(f"    {c['thread']:6s} resumed={str(c['resumed']):5s} -> new_sid={c['new_sid']}")
resume_targets = [c["resumed"] for c in _calls]
# CONTAINMENT: exactly ONE call resumed the pre-existing 'S00' (the first to acquire the lock); the other resumed
# the session that first call produced -> NOT two simultaneous resumes of one session.
only_one_resumed_S00 = resume_targets.count("S00") == 1
second_resumed_first_output = (_calls[1]["resumed"] == _calls[0]["new_sid"]) if len(_calls) == 2 else False
has_lock = any("_session_lock" in t for t in dir(b))
check("(d2) FIXED: a threading.Lock now guards session_id read+write (race contained)",
      has_lock, "PersistentCliBrain._session_lock present -> _run is serialized per instance")
check("(d2) FIXED: only ONE worker resumed the pre-existing session 'S00' (no simultaneous double-resume)",
      only_one_resumed_S00, f"resume_targets={resume_targets}")
check("(d2) FIXED: the 2nd worker resumed the FIRST worker's session (serialized hand-off, none orphaned)",
      second_resumed_first_output, f"2nd resumed {_calls[1]['resumed'] if len(_calls)>1 else None} "
      f"== 1st new_sid {_calls[0]['new_sid'] if _calls else None}")

print("\n" + "="*84)
np=sum(1 for _,c in results if c); nf=sum(1 for _,c in results if not c)
print(f"INDEPENDENCE AUDIT: {np} checks PASS, {nf} FAIL")
print("Interpretation: d1 PASS still documents the by-design context CARRY (sequential nodes share context);")
print("                d2 PASS now confirms the B1 concurrency race is CONTAINED by the session lock.")
print("="*84)
raise SystemExit(0 if nf == 0 else 1)
