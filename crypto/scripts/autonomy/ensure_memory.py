#!/usr/bin/env python3
"""Install-ensure for the MEMORY layer (N5) -- idempotent, LOCAL, graceful (no emoji; cp1252-safe; never hangs).

Makes the richer LOCAL Mem0 memory backend runnable end-to-end on this machine, then PROVES it with a tiny
store+recall ROUND-TRIP through the canonical harness.metaop.memory against the LOCAL ollama embedder + on-disk
qdrant. Designed to be run repeatedly: the second run is a fast no-op. It mirrors ensure_brain.py.

Steps (each prints a [step] line + PASS/FAIL; the script NEVER raises out -- a failed step degrades to a clear
manual instruction and the run continues, then the round-trip is attempted only if preconditions hold):
  1. mem0 importable            -> pip install mem0ai if missing (one attempt; reports the version)
  2. ollama server reachable    -> probe <host>/api/tags (localhost only, 5s; never hangs)
  3. embedder model present      -> `ollama pull nomic-embed-text` ONLY if ollama present + the model is absent
  4. ollama python client        -> pip install ollama if missing (mem0's OllamaEmbedding needs it; else it prompts)
  5. store+recall round-trip     -> write 3 lessons w/ different objectives, recall by a memory query, PASS iff the
                                    task-relevant lesson ranks TOP (proves semantic local retrieval works)

EXIT CODE: 0 iff the round-trip PASSED (the Mem0 backend is live). NON-ZERO if it could not be proven -- BUT this
is NOT a failure of the system: memory DEGRADES GRACEFULLY to the pure-local TF-IDF recall (learnings.py), which
needs none of the above. A non-zero exit means "Mem0 deferred; TF-IDF is carrying memory". The script prints which.

  python scripts/autonomy/ensure_memory.py
  python scripts/autonomy/ensure_memory.py --model nomic-embed-text --host http://localhost:11434
  python scripts/autonomy/ensure_memory.py --no-pull           (skip the embedder pull step)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

# NO-NETWORK + no shared-lock: disable Mem0/posthog telemetry BEFORE `import mem0` anywhere. mem0.memory.telemetry
# reads MEM0_TELEMETRY at ITS import time, so this MUST precede step-1's `import mem0` (else the telemetry qdrant
# store at ~/.mem0/migrations_qdrant is built + locked, and a 2nd Memory instance in-process AlreadyLocked-fails).
os.environ.setdefault("MEM0_TELEMETRY", "False")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_EMBED = os.environ.get("MEM0_EMBED_MODEL", "nomic-embed-text")
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def say(step: str, msg: str) -> None:
    print(f"[ensure-memory:{step}] {msg}", flush=True)


def _pip_install(pkg: str, label: str) -> bool:
    """One pip-install attempt for `pkg`; returns True iff `label` imports afterward. Bounded (600s), never hangs."""
    say(label, f"not importable -- pip install {pkg} (one attempt)...")
    try:
        r = subprocess.run([sys.executable, "-m", "pip", "install", pkg],
                           capture_output=True, text=True, timeout=600,
                           creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
        for ln in ((r.stdout or "").strip().splitlines()[-2:] if r.stdout else []):
            say(label, "  " + ln)
    except subprocess.TimeoutExpired:
        say(label, f"FAIL: pip install timed out. Manual: pip install {pkg}")
        return False
    except Exception as e:
        say(label, f"FAIL: pip install error: {e}")
        return False
    try:
        __import__(label)
        say(label, "importable now -- PASS")
        return True
    except Exception as e:
        say(label, f"FAIL: still not importable: {type(e).__name__}: {e}")
        return False


def ensure_mem0() -> bool:
    """(1) ensure mem0 importable; pip install mem0ai once if missing."""
    try:
        import mem0  # noqa: F401
        say("mem0", f"already installed: mem0=={getattr(mem0, '__version__', '?')} -- PASS")
        return True
    except Exception:
        return _pip_install("mem0ai", "mem0")


def ensure_ollama_client() -> bool:
    """(4) ensure the `ollama` python client is importable -- mem0's OllamaEmbedding imports it and would PROMPT
    (input()) to install it interactively, which would HANG our non-interactive loop. Pre-install it so it never asks."""
    try:
        import ollama  # noqa: F401
        say("ollama-py", "python client present -- PASS")
        return True
    except Exception:
        return _pip_install("ollama", "ollama")


def ollama_tags(host: str) -> list | None:
    try:
        with urllib.request.urlopen(host.rstrip("/") + "/api/tags", timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m.get("name") for m in (data.get("models") or [])]
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    except Exception:
        return None


def ensure_server(host: str) -> tuple[bool, list]:
    """(2) ollama server reachable? (never hangs: 5s probe). Returns (up, model_names)."""
    if shutil.which("ollama"):
        say("ollama", f"binary found: {shutil.which('ollama')}")
    else:
        say("ollama", "binary NOT on PATH. Manual: install Ollama (https://ollama.com), then `ollama serve`.")
    names = ollama_tags(host)
    if names is None:
        say("ollama", f"server NOT reachable at {host}. Manual: `ollama serve`. -- FAIL (will use TF-IDF)")
        return False, []
    say("ollama", f"server up at {host}; {len(names)} model(s): {', '.join(names) or '(none)'} -- PASS")
    return True, names


def ensure_embedder(host: str, model: str, names: list, do_pull: bool) -> bool:
    """(3) pull the embedder model if absent. Guard: only if the ollama binary is present; never hangs (30m cap)."""
    base = model.split(":")[0]
    present = any(n == model or n == model + ":latest" or (n or "").split(":")[0] == base for n in names)
    if present:
        say("embedder", f"'{model}' already present -- PASS (no-op)")
        return True
    if not do_pull:
        say("embedder", f"'{model}' absent and --no-pull set. Manual: `ollama pull {model}` -- SKIP")
        return False
    if not shutil.which("ollama"):
        say("embedder", f"'{model}' absent and ollama binary missing. Manual: `ollama pull {model}` -- SKIP")
        return False
    say("embedder", f"'{model}' absent -- pulling (`ollama pull {model}`; may take a while)...")
    try:
        r = subprocess.run(["ollama", "pull", model], capture_output=True, text=True, timeout=1800,
                           creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
        if r.returncode == 0:
            say("embedder", f"pulled '{model}' -- PASS")
            return True
        say("embedder", f"FAIL: `ollama pull {model}` exit {r.returncode}: {(r.stderr or '').strip()[:200]}")
        return False
    except subprocess.TimeoutExpired:
        say("embedder", f"FAIL: `ollama pull {model}` timed out.")
        return False
    except Exception as e:
        say("embedder", f"FAIL: `ollama pull {model}` error: {e}")
        return False


def round_trip(host: str) -> bool:
    """(5) store 3 lessons w/ different objectives into a TEMP-workspace Mem0, recall by a memory query, PASS iff the
    memory-relevant lesson ranks TOP. Uses the canonical harness.metaop.memory (real code path). Never raises."""
    try:
        from harness.metaop import memory as M
    except Exception as e:
        say("round-trip", f"FAIL: cannot import harness memory: {type(e).__name__}: {e}")
        return False
    tmp = tempfile.mkdtemp(prefix="ensure_mem0_")
    try:
        os.environ["OLLAMA_HOST"] = host  # make the probe + config use the requested host
        if M.backend(workspace=tmp) != "mem0":
            ok, reason = M._preconditions_ok()
            say("round-trip", f"Mem0 backend NOT available ({reason}) -- recall will use TF-IDF (graceful). FAIL")
            return False
        lessons = [
            ("MA crossover needs regime gating to avoid chop on SOL", "find an adaptive moving-average edge on SOL"),
            ("dollar bars clear maker costs better than time bars at 4h", "cost analysis across bar types"),
            ("use TF-IDF cosine over stored objectives for local vector similarity memory retrieval",
             "build a local vector similarity memory for the agent loop"),
        ]
        for text, obj in lessons:
            landed = M.remember(text, objective=obj, channel="ensure_probe", workspace=tmp)
            if not landed:
                say("round-trip", "FAIL: remember() did not land in Mem0 (degraded to TF-IDF mid-probe). FAIL")
                return False
        query = "wire a similarity-based memory retrieval into the loop"
        hits = M._mem0_search(query, 3, "ensure_probe", tmp)
        say("round-trip", f"query={query!r} -> {[(round(s,3), t[:42]) for s, t in hits]}")
        if not hits:
            say("round-trip", "FAIL: Mem0 returned no hits.")
            return False
        top_text = hits[0][1].lower()
        if "tf-idf" in top_text or "vector similarity" in top_text:
            say("round-trip", "PASS -- the memory-relevant lesson ranked TOP via the LOCAL ollama embedder")
            return True
        say("round-trip", "FAIL: top hit was not the memory lesson (semantic ranking off).")
        return False
    except Exception as e:
        say("round-trip", f"FAIL: round-trip raised {type(e).__name__}: {str(e)[:200]}")
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Idempotent local install-ensure for the Mem0 memory backend (N5).")
    ap.add_argument("--model", default=DEFAULT_EMBED, help=f"ollama EMBEDDER model (default {DEFAULT_EMBED})")
    ap.add_argument("--host", default=DEFAULT_HOST, help=f"ollama host (default {DEFAULT_HOST})")
    ap.add_argument("--no-pull", action="store_true", help="do not pull the embedder model if absent")
    args = ap.parse_args()

    say("start", f"embedder={args.model} host={args.host}")
    have_mem0 = ensure_mem0()
    server_up, names = ensure_server(args.host)
    embed_ok = ensure_embedder(args.host, args.model, names, do_pull=not args.no_pull) if server_up else False
    client_ok = ensure_ollama_client() if (have_mem0 and server_up and embed_ok) else False

    if not (have_mem0 and server_up and embed_ok and client_ok):
        say("result", "Mem0 PRECONDITIONS INCOMPLETE -- memory degrades to the pure-local TF-IDF recall "
                      "(learnings.py), which needs none of the above. This is SAFE, just not the richer backend. FAIL")
        return 1
    ok = round_trip(args.host)
    print(("[ensure-memory] PASS -- Mem0 LOCAL backend live (ollama embedder + on-disk qdrant); "
           "recall is semantic + TF-IDF-fused") if ok else
          "[ensure-memory] FAIL -- Mem0 round-trip not proven; recall falls back to pure-local TF-IDF (SAFE)",
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
