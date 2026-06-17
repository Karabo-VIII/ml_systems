#!/usr/bin/env python3
"""Install-ensure for the BRAIN layer -- idempotent, LOCAL, graceful (no emoji; cp1252-safe; never hangs/prompts).

Makes the LiteLLM-routed brain runnable end-to-end on this machine, then PROVES it with a tiny live completion
through LiteLLMBrain against the LOCAL ollama server. Designed to be run repeatedly: the second run is a fast no-op.

Steps (each prints a [step] line + PASS/FAIL; the script NEVER raises out -- a failed step degrades to a clear
manual instruction and the run continues so you always get the full picture):
  1. litellm importable           -> pip install if missing (one attempt; reports the version)
  2. ollama binary + server       -> probe `<host>/api/tags` (no network beyond localhost)
  3. configured model present      -> `ollama pull <model>` ONLY if ollama is present + the model is absent
  4. live LiteLLMBrain.decide()    -> a real local completion; PASS iff a dict comes back

Exit code 0 iff the final live test PASSED (the overseer's RWYB anchor); non-zero otherwise. The script is safe to
re-run; it does no destructive work and prompts for nothing.

  python scripts/autonomy/ensure_brain.py                 (auto-selects the most capable pulled model that fits)
  python scripts/autonomy/ensure_brain.py --model qwen2.5-coder:7b --host http://localhost:11434
  python scripts/autonomy/ensure_brain.py --no-pull       (skip the model pull step)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

# import the canonical harness brain (repo root on sys.path -> `harness.metaop.brain`).
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Local-brain capability ladder, MOST-CAPABLE FIRST, with the GPU VRAM (MiB) each needs to run as the SINGLE
# resident model (4-bit weights + KV cache @ 4096 ctx + overhead). "Most capable we can run locally" (user mandate
# 2026-06-12) = the top of this ladder whose `need_mib` fits the GPU's TOTAL VRAM. NOTE we gate on TOTAL, not FREE,
# because the brain owns the GPU (only ONE model should be resident); two models co-loaded is the contention bug
# this selector exists to avoid -- standardize on one and ollama unloads the rest after keep-alive.
MODEL_LADDER = [
    ("qwen2.5-coder:7b", 5600),   # ~4.7GB weights + KV -> needs an 8GB-class GPU; the target on this RTX 4060
    ("qwen2.5-coder:3b", 2600),   # ~1.9GB weights + KV -> the safe fallback on a <6.5GB GPU
]
_VRAM_RESERVE_MIB = 0  # already folded into need_mib above


def nvidia_total_mib() -> int | None:
    """TOTAL GPU VRAM in MiB via nvidia-smi, or None if no NVIDIA GPU / tool absent (never raises, 5s cap)."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5,
                           creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
        if r.returncode == 0 and r.stdout.strip():
            return max(int(x) for x in r.stdout.strip().splitlines())  # largest GPU if multiple
    except Exception:
        pass
    return None


def best_local_model(host: str, names: list | None = None) -> tuple[str, str]:
    """Pick the MOST CAPABLE local model that (a) is pulled and (b) fits this GPU's TOTAL VRAM. An explicit
    OLLAMA_MODEL env var always wins (operator override). Returns (model, reason). Never raises -- on any
    uncertainty it degrades toward the smallest known-safe model so the loop always has a brain."""
    env = os.environ.get("OLLAMA_MODEL")
    if env:
        return env, "OLLAMA_MODEL env override"
    if names is None:
        names = ollama_tags(host) or []
    pulled = {(n or "").split(":")[0] + ":" + ((n or "").split(":")[1] if ":" in (n or "") else "latest"): n
              for n in names}
    have = lambda m: any(n == m or (n or "").startswith(m.split(":")[0] + ":") for n in names)
    total = nvidia_total_mib()
    for model, need in MODEL_LADDER:        # most-capable first
        if not have(model):
            continue
        if total is None or total >= need:
            why = (f"fits GPU total {total} MiB >= {need}" if total is not None
                   else "no GPU probe -> assume it fits (CPU/offload OK)")
            return model, why
    # nothing on the ladder is pulled (or none fit) -> smallest ladder entry as a target to pull, else legacy default
    return MODEL_LADDER[-1][0], "no ladder model pulled/fitting -> smallest as pull-target"


DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def say(step: str, msg: str) -> None:
    print(f"[ensure-brain:{step}] {msg}", flush=True)


def litellm_version() -> str | None:
    """Robust litellm version (litellm 1.88.0 does NOT expose litellm.__version__; use importlib.metadata)."""
    try:
        from importlib.metadata import version
        return version("litellm")
    except Exception:
        try:
            import litellm  # noqa: F401
            return getattr(litellm, "__version__", "unknown")
        except Exception:
            return None


def ensure_litellm() -> bool:
    """(a) ensure litellm importable; pip install once if missing. Returns True iff importable at the end."""
    v = litellm_version()
    if v is not None:
        try:
            import litellm  # noqa: F401  -- confirm it actually imports (metadata can lag a broken install)
            say("litellm", f"already installed: litellm=={v} -- PASS")
            return True
        except Exception as e:
            say("litellm", f"metadata says {v} but import failed ({type(e).__name__}); reinstalling...")
    say("litellm", "not importable -- pip install litellm (one attempt)...")
    try:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "litellm"],
                           capture_output=True, text=True, timeout=600,
                           creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
        tail = (r.stdout or "").strip().splitlines()[-3:] if r.stdout else []
        for ln in tail:
            say("litellm", "  " + ln)
    except subprocess.TimeoutExpired:
        say("litellm", "FAIL: pip install timed out. Manual: .venv/Scripts/python.exe -m pip install litellm")
        return False
    except Exception as e:
        say("litellm", f"FAIL: pip install error: {e}. Manual: pip install litellm")
        return False
    v = litellm_version()
    try:
        import litellm  # noqa: F401
        say("litellm", f"installed: litellm=={v} -- PASS")
        return True
    except Exception as e:
        say("litellm", f"FAIL: still not importable after install: {type(e).__name__}: {e}")
        return False


def ollama_binary() -> str | None:
    return shutil.which("ollama")


def ollama_tags(host: str) -> list | None:
    """Return the list of model names from <host>/api/tags, or None if the server is unreachable (no hang)."""
    try:
        req = urllib.request.Request(host.rstrip("/") + "/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m.get("name") for m in (data.get("models") or [])]
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    except Exception:
        return None


def ensure_ollama_server(host: str) -> tuple[bool, list]:
    """(b) check the ollama binary + server. Returns (server_up, model_names). Never hangs (5s probe)."""
    binp = ollama_binary()
    if binp:
        say("ollama", f"binary found: {binp}")
    else:
        say("ollama", "binary NOT on PATH. Manual: install Ollama (https://ollama.com), then `ollama serve`.")
    names = ollama_tags(host)
    if names is None:
        say("ollama", f"server NOT reachable at {host}. Manual: start it with `ollama serve`. -- FAIL")
        return False, []
    say("ollama", f"server up at {host}; {len(names)} model(s): {', '.join(names) if names else '(none)'} -- PASS")
    return True, names


def ensure_model(host: str, model: str, names: list, do_pull: bool) -> bool:
    """(c) pull the configured model if absent. Guard: only if the ollama binary is present; else print a manual
    instruction (do NOT hang). Returns True iff the model is present at the end."""
    present = any(n == model or n == model + ":latest" or (n or "").split(":")[0] == model.split(":")[0]
                 for n in names)
    if present:
        say("model", f"'{model}' already present -- PASS (no-op)")
        return True
    if not do_pull:
        say("model", f"'{model}' absent and --no-pull set. Manual: `ollama pull {model}` -- SKIP")
        return False
    if not ollama_binary():
        say("model", f"'{model}' absent and ollama binary missing. Manual: install Ollama, then "
                     f"`ollama pull {model}` -- SKIP")
        return False
    say("model", f"'{model}' absent -- pulling (`ollama pull {model}`; this can take a while)...")
    try:
        r = subprocess.run(["ollama", "pull", model], capture_output=True, text=True, timeout=1800,
                           creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
        if r.returncode == 0:
            say("model", f"pulled '{model}' -- PASS")
            return True
        say("model", f"FAIL: `ollama pull {model}` exit {r.returncode}: {(r.stderr or '').strip()[:200]}")
        return False
    except subprocess.TimeoutExpired:
        say("model", f"FAIL: `ollama pull {model}` timed out. Manual: run it yourself.")
        return False
    except Exception as e:
        say("model", f"FAIL: `ollama pull {model}` error: {e}")
        return False


def live_test(host: str, model: str) -> bool:
    """(d) run a tiny test completion through LiteLLMBrain against ollama. PASS iff a dict comes back (parsed JSON
    from the live local model). Prints PASS/FAIL. Never raises."""
    try:
        from harness.metaop.brain import make_brain, LiteLLMBrain
    except Exception as e:
        say("live", f"FAIL: cannot import harness brain: {type(e).__name__}: {e}")
        return False
    try:
        b = make_brain("litellm", model="ollama/" + model)
        if not isinstance(b, LiteLLMBrain):
            say("live", f"NOTE: make_brain returned {b.name} (litellm absent?) -- proceeding with it anyway")
        os.environ.setdefault("LITELLM_API_BASE", host)
        out = b.decide("plan", {"objective": "ensure-brain smoke: return a tiny frontier"})
        ok = isinstance(out, dict) and not out.get("_error")
        say("live", f"brain={b.name} model=ollama/{model} decide()-> {json.dumps(out, default=str)[:300]}")
        say("live", "PASS" if ok else "FAIL: brain returned an _error or non-dict")
        return ok
    except Exception as e:  # belt-and-suspenders: the brain promises not to raise, but never let this script crash
        say("live", f"FAIL: live test raised {type(e).__name__}: {str(e)[:200]}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Idempotent local install-ensure for the LiteLLM-routed brain.")
    ap.add_argument("--model", default=None,
                    help="ollama model id; default = auto-select the most capable pulled model that fits this GPU "
                         "(see best_local_model / MODEL_LADDER). Pass an explicit id to force one.")
    ap.add_argument("--host", default=DEFAULT_HOST, help=f"ollama host (default {DEFAULT_HOST})")
    ap.add_argument("--no-pull", action="store_true", help="do not pull the model if absent")
    args = ap.parse_args()

    say("start", f"host={args.host}")
    have_litellm = ensure_litellm()
    server_up, names = ensure_ollama_server(args.host)
    if args.model:
        model = args.model
        say("model-select", f"forced by --model: {model}")
    else:
        model, why = best_local_model(args.host, names)
        say("model-select", f"auto-selected '{model}' ({why})")
    model_ok = ensure_model(args.host, model, names, do_pull=not args.no_pull) if server_up else False

    if not (have_litellm and server_up and model_ok):
        say("result", "PRECONDITIONS INCOMPLETE -- see manual instructions above; skipping live test. FAIL")
        return 1
    ok = live_test(args.host, model)
    print(("[ensure-brain] PASS -- LiteLLMBrain answered live against ollama/" + model)
          if ok else "[ensure-brain] FAIL -- live test did not pass (see above)", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
