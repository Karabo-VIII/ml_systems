"""FALSIFIER (b) REAL negative control with the live `claude` CLI (no fakes).

Three real claude -p calls:
  1. FRESH  : plant a secret token in a new session; capture its session_id.
  2. RESUME : --resume that session, ask for the token  -> EXPECT recall (continuity).
  3. CONTROL: a BRAND-NEW fresh session (no --resume), ask for the token -> EXPECT NO recall.
Genuine persistence requires (2) recalls AND (3) does not. If (3) also recalls, (2) is coincidence.
No emoji (cp1252).
"""
from __future__ import annotations
import json, subprocess, sys
from scripts.autonomy.metaop.brain import find_claude

EXE = find_claude()
TOKEN = "QUOKKA-49162-VX"
TIMEOUT = 240

def run(prompt, resume=None):
    cmd = [EXE, "-p", prompt, "--output-format", "json"]
    if resume:
        cmd += ["--resume", resume]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        return {"_error": "timeout"}, ""
    raw = (r.stdout or "").strip()
    try:
        env = json.loads(raw)
    except Exception as e:
        return {"_error": f"parse {e}", "_raw": raw[:200]}, raw[:200]
    return env, env.get("result", "")

print(f"EXE={EXE}\nTOKEN={TOKEN}\n" + "="*70)

print("[1] FRESH: planting the token ...")
env1, res1 = run(f"Remember this secret token exactly: {TOKEN}. "
                 f"Do not use any tools. Reply with only the word OK.")
sid = env1.get("session_id")
print(f"    session_id={sid}  result={res1!r}  err={env1.get('_error')}")
if not sid:
    print("RESULT: could not establish a session (no session_id) -- cannot run negative control")
    sys.exit(2)

print("[2] RESUME: asking the SAME session for the token ...")
env2, res2 = run("What was the secret token I told you earlier? "
                 "Do not use any tools. Reply with ONLY the token, nothing else.", resume=sid)
print(f"    resumed_session_id={env2.get('session_id')}  result={res2!r}  err={env2.get('_error')}")
resume_recalls = TOKEN in (res2 or "")

print("[3] CONTROL: a BRAND-NEW fresh session, same question (must NOT know the token) ...")
env3, res3 = run("What was the secret token I told you earlier? "
                 "Do not use any tools. Reply with ONLY the token; if you were not told one, reply NONE.")
print(f"    control_session_id={env3.get('session_id')}  result={res3!r}  err={env3.get('_error')}")
control_recalls = TOKEN in (res3 or "")

print("="*70)
print(f"resume_recalls_token  = {resume_recalls}   (want True)")
print(f"control_recalls_token = {control_recalls}  (want False)")
genuine = resume_recalls and not control_recalls
print(f"GENUINE PERSISTENCE (resume recalls AND fresh control does not): {genuine}")
sys.exit(0 if genuine else 1)
