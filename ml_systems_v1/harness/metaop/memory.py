"""Harness MEMORY -- a richer LOCAL memory backend (Mem0) BEHIND the existing learnings interface, with the
pure-local TF-IDF recall as a GUARANTEED fallback.

WHY: learnings.py already gives us monotonic, channel-keyed compounding memory + TF-IDF task-similarity recall
(`similar_for_plan`), which is robust and dependency-light. Mem0 ADDS a vector-embedding retrieval layer (semantic
similarity via a real embedder) that can surface a relevant past lesson even when the wording differs from the
query -- something bag-of-words TF-IDF misses. This module is the thin seam that uses Mem0 WHEN it is fully local +
configured + reachable, and otherwise DELEGATES to learnings (TF-IDF). It is the single import the loop touches.

FULLY LOCAL contract (NO cloud, NO OpenAI key, NO network beyond localhost):
  - LLM       : ollama (qwen2.5-coder:3b @ http://localhost:11434)   -- only used for Mem0's fact-extraction; we
                store verbatim (infer=False) so the LLM is NOT on the hot path of remember()/recall().
  - Embedder  : ollama (nomic-embed-text @ http://localhost:11434)   -- the semantic-retrieval workhorse.
  - Store     : qdrant in EMBEDDED on-disk mode under the harness WORKSPACE (runs/autonomy/mem0 for the crypto
                consumer; .harness_runs/mem0 for an agnostic run). No server, no port, no network.

BEST-EFFORT / NEVER-RAISE: every Mem0 path is wrapped so ANY error (mem0 absent, ollama down, embedder missing,
store I/O, API drift) DEGRADES to the TF-IDF learnings path. A memory layer must never wedge or crash the loop --
the #1 rule is "memory degrades gracefully to TF-IDF". No emoji (Windows cp1252).

PUBLIC INTERFACE (stable):
  remember(text, meta=None, channel=..., workspace=...) -> bool   # True iff it also landed in Mem0
  recall(objective, k=3, channel=..., workspace=...)     -> str    # prompt-ready digest, ALWAYS non-empty
  backend(workspace=...)                                 -> str    # "mem0" | "tfidf" (which path recall would use)
  available(workspace=...)                               -> bool   # is the Mem0 local backend usable right now?

remember() ALWAYS writes to learnings (the durable JSONL of record, the TF-IDF fallback's source of truth) AND,
best-effort, ALSO to Mem0. recall() prefers Mem0's semantic hits, FUSED with the TF-IDF digest, and falls back to
TF-IDF-only when Mem0 is unavailable -- so the caller gets the best of both and never an empty result.
"""
from __future__ import annotations

import os
import urllib.error
import urllib.request

# FULLY-LOCAL + NO-NETWORK contract: disable Mem0's (and the bundled posthog's) phone-home telemetry BEFORE mem0 is
# ever imported. This is load-bearing twice over: (1) it honors the no-network rule, and (2) with telemetry ON Mem0
# builds a SECOND, SHARED qdrant store at ~/.mem0/migrations_qdrant -- two Memory instances in one process then
# collide on that shared file lock (qdrant local AlreadyLocked). Telemetry OFF removes that store + the collision.
os.environ.setdefault("MEM0_TELEMETRY", "False")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from . import learnings
from .config import workspace_root

# ----------------------------------------------------------------------------------------------------------------
# Local, no-network configuration. The embedder is the part that MUST exist for Mem0 to add value; if it is absent
# we fall straight back to TF-IDF (never hang on a pull, never call out to a cloud).
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MEM0_LLM_MODEL = os.environ.get("MEM0_LLM_MODEL", os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:3b"))
MEM0_EMBED_MODEL = os.environ.get("MEM0_EMBED_MODEL", "nomic-embed-text")
MEM0_EMBED_DIMS = int(os.environ.get("MEM0_EMBED_DIMS", "768"))  # nomic-embed-text -> 768
# a stable scope id so all harness lessons share one Mem0 namespace (Mem0 needs a user/agent/run id on every op).
MEM0_SCOPE = os.environ.get("MEM0_SCOPE", "harness")
_PROBE_TIMEOUT = 4  # seconds; the localhost reachability probe must NEVER hang the loop

# one Memory instance per (workspace, channel) -- building it re-opens the embedded store, so we cache.
_CACHE: dict = {}
_DISABLED = os.environ.get("HARNESS_MEM0_DISABLE", "").strip().lower() in ("1", "true", "yes", "on")


def _close_cached() -> None:
    """Close cached embedded-qdrant clients at interpreter exit. Cosmetic-only: without it qdrant's __del__ fires
    AFTER sys.meta_path is torn down and prints a harmless 'Exception ignored ... Python is likely shutting down'
    traceback. Closing here pre-empts that noise. Wrapped in try/except -- a close error must never crash teardown."""
    for m in list(_CACHE.values()):
        try:
            vs = getattr(m, "vector_store", None)
            client = getattr(vs, "client", None)
            if client is not None and hasattr(client, "close"):
                client.close()
        except Exception:
            pass
    _CACHE.clear()


import atexit as _atexit  # noqa: E402  -- register the teardown-noise suppressor once at import
_atexit.register(_close_cached)


def mem0_dir(workspace: str | None = None):
    """The LOCAL on-disk store root under the harness workspace (runs/autonomy/mem0 or .harness_runs/mem0)."""
    d = workspace_root(workspace) / "mem0"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ollama_models(host: str = OLLAMA_HOST) -> list | None:
    """Model names from <host>/api/tags, or None if the local server is unreachable. Localhost-only, 4s cap."""
    try:
        with urllib.request.urlopen(host.rstrip("/") + "/api/tags", timeout=_PROBE_TIMEOUT) as resp:
            import json
            data = json.loads(resp.read().decode("utf-8"))
        return [m.get("name") for m in (data.get("models") or [])]
    except (urllib.error.URLError, TimeoutError, OSError, Exception):
        return None


def _embedder_present(names: list | None) -> bool:
    """True iff the configured embedder model is pulled (match with/without an explicit :tag)."""
    if not names:
        return False
    base = MEM0_EMBED_MODEL.split(":")[0]
    return any(n == MEM0_EMBED_MODEL or n == MEM0_EMBED_MODEL + ":latest"
              or (n or "").split(":")[0] == base for n in names)


def _preconditions_ok() -> tuple[bool, str]:
    """Cheap, no-hang gate: mem0 importable + ollama server up + the embedder model pulled. Returns (ok, reason)."""
    if _DISABLED:
        return False, "disabled via HARNESS_MEM0_DISABLE"
    try:
        import mem0  # noqa: F401
    except Exception as e:
        return False, f"mem0 not importable ({type(e).__name__})"
    names = _ollama_models()
    if names is None:
        return False, f"ollama server not reachable at {OLLAMA_HOST}"
    if not _embedder_present(names):
        return False, f"embedder '{MEM0_EMBED_MODEL}' not pulled (have: {names})"
    return True, "ok"


def _config(workspace: str | None, channel: str) -> dict:
    """Fully-local Mem0 config: ollama LLM + ollama embedder + EMBEDDED on-disk qdrant under the workspace."""
    root = mem0_dir(workspace)
    # per-channel collection + store dir so different lanes stay isolated (mirrors learnings' channel model).
    safe_chan = "".join(c if (c.isalnum() or c in "_-") else "_" for c in (channel or "default"))
    return {
        "llm": {"provider": "ollama", "config": {"model": MEM0_LLM_MODEL,
                "ollama_base_url": OLLAMA_HOST, "temperature": 0.1}},
        "embedder": {"provider": "ollama", "config": {"model": MEM0_EMBED_MODEL,
                "ollama_base_url": OLLAMA_HOST}},
        "vector_store": {"provider": "qdrant", "config": {
                "path": str(root / f"qdrant_{safe_chan}"),
                "collection_name": f"harness_{safe_chan}",
                "on_disk": True, "embedding_model_dims": MEM0_EMBED_DIMS}},
        "history_db_path": str(root / f"history_{safe_chan}.db"),
    }


def _get_memory(workspace: str | None, channel: str):
    """Return a cached Mem0 Memory for (workspace, channel), or None if it cannot be built locally. Never raises."""
    if _DISABLED:
        return None
    key = (str(workspace_root(workspace)), channel or "default")
    if key in _CACHE:
        return _CACHE[key]  # may be a Memory or the None sentinel (cached negative -> don't re-probe every call)
    ok, _reason = _preconditions_ok()
    if not ok:
        _CACHE[key] = None
        return None
    try:
        # quiet mem0's chatty (and cp1252-unsafe) INFO logging about optional NLP extras (spaCy/fastembed/BM25)
        # that we deliberately do not use -- we store VERBATIM (infer=False), so those features are irrelevant.
        import logging as _logging
        for _name in ("mem0", "mem0.memory", "mem0.vector_stores"):
            _logging.getLogger(_name).setLevel(_logging.ERROR)
        from mem0 import Memory
        m = Memory.from_config(_config(workspace, channel))
        _CACHE[key] = m
        return m
    except Exception:
        _CACHE[key] = None  # construction failed -> cache the miss, fall back to TF-IDF from here on
        return None


def available(workspace: str | None = None, channel: str = "default") -> bool:
    """Is the LOCAL Mem0 backend usable right now (built + reachable)? Used by ensure_memory + diagnostics."""
    return _get_memory(workspace, channel) is not None


def backend(workspace: str | None = None, channel: str = "default") -> str:
    """Which path recall() would take RIGHT NOW: 'mem0' (semantic) or 'tfidf' (the guaranteed fallback)."""
    return "mem0" if available(workspace, channel) else "tfidf"


def remember(text: str, meta: dict | None = None, thread: str = "mem", objective: str = "", cycle: int = 0,
             channel: str = "default", workspace: str | None = None) -> bool:
    """Persist a lesson. ALWAYS records to learnings (the durable JSONL that the TF-IDF fallback reads), and ALSO,
    best-effort, into Mem0's local vector store for semantic recall. Returns True iff it also landed in Mem0.
    Never raises into the loop -- a Mem0 failure just means TF-IDF-only for this item."""
    if not text or not str(text).strip():
        return False
    # 1) durable JSONL (the source of truth + the TF-IDF fallback's corpus). objective is stored so similarity works.
    learnings.record(str(text), thread, objective or (meta or {}).get("objective", ""), cycle,
                     channel=channel, workspace=workspace)
    # 2) Mem0 vector store (best-effort, additive). infer=False -> store VERBATIM (no LLM rewrite on the hot path).
    m = _get_memory(workspace, channel)
    if m is None:
        return False
    try:
        md = {"thread": thread, "channel": channel}
        if objective:
            md["objective"] = str(objective)[:160]
        if isinstance(meta, dict):
            md.update({k: v for k, v in meta.items() if isinstance(k, str)})
        m.add(str(text), user_id=MEM0_SCOPE, metadata=md, infer=False)
        return True
    except Exception:
        return False  # store hiccup -> the lesson is still safe in learnings; recall falls back to TF-IDF


def _mem0_search(objective: str, k: int, channel: str, workspace: str | None) -> list:
    """Top-k semantic hits from Mem0 for THIS objective, or [] if Mem0 is unavailable / errors. Each hit is
    (score, text). Handles the v2 API (filters={'user_id': ...}) and tolerates result-shape drift. Never raises."""
    m = _get_memory(workspace, channel)
    if m is None:
        return []
    try:
        res = m.search(str(objective), filters={"user_id": MEM0_SCOPE}, limit=max(1, k))
        rows = res.get("results", res) if isinstance(res, dict) else res
        out = []
        for r in (rows or []):
            if not isinstance(r, dict):
                continue
            txt = r.get("memory") or r.get("text") or ""
            if txt:
                out.append((float(r.get("score", 0.0) or 0.0), str(txt)))
        return out[:k]
    except Exception:
        return []


def recall(objective: str, k: int = 3, channel: str = "default", workspace: str | None = None,
           across_channels: bool = True) -> str:
    """Prompt-ready digest of prior lessons most relevant to THIS objective. Prefers Mem0's SEMANTIC hits and FUSES
    them with the TF-IDF digest (so wording-divergent matches Mem0 finds AND the bag-of-words matches TF-IDF finds
    both surface). ALWAYS returns a non-empty string: with Mem0 down it is exactly the TF-IDF `similar_for_plan`
    output, so the caller's behavior is unchanged on the fallback path. Never raises into the loop."""
    obj = str(objective or "").strip()
    if not obj:
        return "(no objective given for memory recall)"
    # TF-IDF digest is ALWAYS computed -- it is the guaranteed floor (and dedups against the Mem0 hits below).
    tfidf = ""
    try:
        tfidf = learnings.similar_for_plan(obj, k=k, channel=channel, workspace=workspace,
                                           across_channels=across_channels)
    except Exception:
        tfidf = "(task-similarity recall unavailable)"
    mem_hits = _mem0_search(obj, k, channel, workspace)
    if not mem_hits:
        return tfidf  # Mem0 unavailable/empty -> behave EXACTLY as the existing TF-IDF recall (fallback proof).
    lines = [f"- (mem0 sim={s:.2f}) {t}" for s, t in mem_hits]
    header = ("SEMANTIC MEMORY HITS (Mem0 local vector recall, retrieved by embedding-similarity to THIS objective "
              "-- reuse these even if worded differently; do NOT re-derive them):")
    out = header + "\n" + "\n".join(lines)
    # fuse: append the TF-IDF block too (different recall basis), unless it's just a 'nothing found' note.
    if tfidf and not tfidf.startswith("(") and tfidf not in out:
        out += "\n" + tfidf
    return out


if __name__ == "__main__":
    # tiny self-demo (uses the default .harness_runs workspace): which backend, a store + a recall.
    print("backend:", backend())
    remember("use TF-IDF cosine over stored objectives for local vector similarity memory retrieval",
             objective="build a local vector similarity memory for the agent loop", channel="demo")
    remember("MA crossover needs regime gating to avoid chop on SOL",
             objective="find an adaptive moving-average edge on SOL", channel="demo")
    print(recall("wire a similarity-based memory retrieval into the loop", k=2, channel="demo"))
