"""Harness TOOLS -- the executor that lets a worker DO real work (shell / python / read / write / list).

Project-agnostic version of the original metaop tools. Two safety fences, ALWAYS enforced:
  HARD_DENY      -- shell commands that are irreversible/destructive (rm -rf /, force-push, mkfs, sudo, ...)
  HARD_FILE_DENY -- control-surface files a worker must never write (.git internals, .env, ...)

The original was wired to a crypto-repo permission_policy.json + a "loops never commit" fence. Here the fence is
self-contained (no external policy file required), but you can still SUPPLY extra deny patterns at construction
(`extra_cmd_deny=` / `extra_file_deny=`) to tighten it per domain -- e.g. add a git-commit fence if you want the
overseer to own commits. The build cwd is injectable (defaults to the harness build_cwd). No emoji (cp1252).
"""
from __future__ import annotations

import html as _html
import json
import os
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .config import build_cwd, scratch_dir

# SSRF guard for fetch_url/web_search: a worker must not reach internal services (ollama, localhost, cloud metadata,
# private ranges). Research tools hit the PUBLIC web only.
_PRIVATE_HOST = re.compile(
    r"^(localhost|127\.|0\.0\.0\.0|10\.|192\.168\.|169\.254\.|172\.(1[6-9]|2\d|3[01])\.|\[?::1\]?|metadata\.)",
    re.IGNORECASE)

# HARD shell fence -- irreversible / destructive ops are ALWAYS denied regardless of caller config.
HARD_DENY = [
    r"rm\s+-rf\s+[/~]", r"\brmdir\s+/s\b", r"\bdel\s+/[sq]\b",
    r"git\s+push\s+\S*\s*--force", r"\bgit\s+push\s+--force\b", r"git\s+reset\s+--hard",
    r"(^|\s)sudo\s", r"(^|\s)dd\s+if=", r"\bmkfs\b", r"format\s+c:",
    r"\bshutdown\b", r"\breboot\b", r":\(\)\s*\{",  # fork bomb
]

# HARD file fence -- control surfaces a worker must never overwrite.
HARD_FILE_DENY = [r"\.git[\\/]", r"\.env\b", r"\bid_rsa\b", r"\.ssh[\\/]"]


def _blocked(text: str, patterns: list) -> str | None:
    for pat in patterns:
        try:
            if re.search(pat, text):
                return pat
        except re.error:
            continue
    return None


class Tools:
    """Tool surface for a worker. cwd = where commands run + paths resolve (the target project)."""

    def __init__(self, cwd: Path | str | None = None, workspace: str | None = None, timeout: int = 300,
                 extra_cmd_deny: list | None = None, extra_file_deny: list | None = None,
                 allow_web: bool = True, web_timeout: int = 20):
        self.root = Path(cwd) if cwd else build_cwd()
        self.workspace = workspace
        self.timeout = timeout
        self.cmd_deny = HARD_DENY + list(extra_cmd_deny or [])
        self.file_deny = HARD_FILE_DENY + list(extra_file_deny or [])
        self.allow_web = allow_web      # research tools (web_search/fetch_url); set False for offline/sandboxed runs
        self.web_timeout = web_timeout

    # -- shell ---------------------------------------------------------------
    def run_shell(self, command: str) -> dict:
        hit = _blocked(command, self.cmd_deny)
        if hit:
            return {"ok": False, "tool": "run_shell", "error": f"DENIED by safety fence /{hit}/"}
        try:
            # CREATE_NO_WINDOW (Windows): keep shell commands headless -- no console window flashes / steals focus
            # when the worker runs tools repeatedly (the user's "terminal pops up + pauses typing" annoyance).
            _nw = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            r = subprocess.run(command, shell=True, cwd=str(self.root), capture_output=True,
                               text=True, timeout=self.timeout, creationflags=_nw)
            out = (r.stdout or "")[-6000:] + (("\n[stderr]\n" + r.stderr[-2000:]) if r.returncode and r.stderr else "")
            return {"ok": r.returncode == 0, "tool": "run_shell", "exit": r.returncode, "output": out.strip()}
        except subprocess.TimeoutExpired:
            return {"ok": False, "tool": "run_shell", "error": f"timeout after {self.timeout}s"}
        except Exception as e:
            return {"ok": False, "tool": "run_shell", "error": str(e)}

    def run_python(self, code: str) -> dict:
        # write to a scratch file under the harness workspace and execute (safer than -c for multiline)
        tmp = scratch_dir(self.workspace) / "_snippet.py"
        tmp.write_text(code, encoding="utf-8")
        res = self.run_shell(f'python "{tmp}"')
        res["tool"] = "run_python"
        return res

    # -- files ---------------------------------------------------------------
    def read_file(self, path: str, max_chars: int = 8000) -> dict:
        fp = (self.root / path).resolve()
        try:
            return {"ok": True, "tool": "read_file", "output": fp.read_text(encoding="utf-8", errors="replace")[:max_chars]}
        except Exception as e:
            return {"ok": False, "tool": "read_file", "error": str(e)}

    def write_file(self, path: str, content: str) -> dict:
        hit = _blocked(str(path), self.file_deny)
        if hit:
            return {"ok": False, "tool": "write_file", "error": f"DENIED writing protected path /{hit}/"}
        fp = (self.root / path).resolve()
        try:
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content, encoding="utf-8")
            return {"ok": True, "tool": "write_file", "output": f"wrote {len(content)} chars to {path}"}
        except Exception as e:
            return {"ok": False, "tool": "write_file", "error": str(e)}

    def list_dir(self, path: str = ".") -> dict:
        fp = (self.root / path).resolve()
        try:
            return {"ok": True, "tool": "list_dir", "output": "\n".join(sorted(os.listdir(fp))[:200])}
        except Exception as e:
            return {"ok": False, "tool": "list_dir", "error": str(e)}

    # -- research (PUBLIC web; gives a brain WITHOUT native web access -- e.g. a local Ollama model -- the ability
    #    to search + read pages. Pure stdlib urllib, no API key, no deps. SSRF-guarded. Never raises into the loop) --
    def _http_get(self, url: str) -> str:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (harness-research-tool)"})
        with urllib.request.urlopen(req, timeout=self.web_timeout) as r:
            return r.read().decode("utf-8", "replace")

    def web_search(self, query: str, max_results: int = 5) -> dict:
        """Public-web search. Backends (graceful, no hang): (1) BRAVE_API_KEY -> Brave Search API (real general web
        results, 2000/mo free tier); else (2) DuckDuckGo Instant-Answer JSON API (no key, no bot-block, but only
        entity abstracts + related topics -- not general web). The DDG *HTML* scraper is intentionally NOT used (it
        bot-blocks). For deep/general search, set BRAVE_API_KEY. Returns title + url + snippet lines."""
        if not self.allow_web:
            return {"ok": False, "tool": "web_search", "error": "web disabled (allow_web=False)"}
        key = os.environ.get("BRAVE_API_KEY") or os.environ.get("BRAVE_SEARCH_API_KEY")
        try:
            if key:  # (1) real general web search
                req = urllib.request.Request(
                    "https://api.search.brave.com/res/v1/web/search?" +
                    urllib.parse.urlencode({"q": query, "count": max_results}),
                    headers={"Accept": "application/json", "X-Subscription-Token": key})
                with urllib.request.urlopen(req, timeout=self.web_timeout) as r:
                    data = json.loads(r.read().decode("utf-8", "replace"))
                out = [f"{i+1}. {w.get('title','')}\n   {w.get('url','')}\n   {(w.get('description','') or '')[:240]}"
                       for i, w in enumerate((data.get("web", {}).get("results", []) or [])[:max_results])]
                if out:
                    return {"ok": True, "tool": "web_search", "output": "\n".join(out)}
            # (2) no-key fallback: DDG Instant-Answer JSON (entities/definitions; never bot-blocks)
            ia = json.loads(self._http_get("https://api.duckduckgo.com/?" + urllib.parse.urlencode(
                {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"})))
            out = []
            if ia.get("AbstractText"):
                out.append(f"1. {ia.get('Heading','')}\n   {ia.get('AbstractURL','')}\n   {ia['AbstractText'][:300]}")
            for rt in (ia.get("RelatedTopics", []) or []):
                if isinstance(rt, dict) and rt.get("Text") and rt.get("FirstURL") and len(out) < max_results:
                    out.append(f"{len(out)+1}. {rt['Text'][:80]}\n   {rt['FirstURL']}\n   {rt['Text'][:240]}")
            if out:
                hint = "" if key else "\n[note: no BRAVE_API_KEY -> entity-only results; set it for full web search; use fetch_url to read any page]"
                return {"ok": True, "tool": "web_search", "output": "\n".join(out) + hint}
            return {"ok": True, "tool": "web_search",
                    "output": f"no instant-answer for '{query}'. Set BRAVE_API_KEY for general web search, "
                              "or use fetch_url(url) if you know a relevant page."}
        except Exception as e:
            return {"ok": False, "tool": "web_search", "error": f"{type(e).__name__}: {str(e)[:200]}"}

    def fetch_url(self, url: str, max_chars: int = 6000) -> dict:
        """Fetch a PUBLIC http(s) URL and return its text (HTML stripped). SSRF-guarded (no localhost/private/internal)."""
        if not self.allow_web:
            return {"ok": False, "tool": "fetch_url", "error": "web disabled (allow_web=False)"}
        try:
            p = urllib.parse.urlparse(url)
            if p.scheme not in ("http", "https"):
                return {"ok": False, "tool": "fetch_url", "error": "only http/https URLs allowed"}
            if not p.hostname or _PRIVATE_HOST.match(p.hostname):
                return {"ok": False, "tool": "fetch_url", "error": "SSRF guard: internal/private host refused"}
            page = self._http_get(url)
            txt = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", page, flags=re.S | re.I)
            txt = _html.unescape(re.sub(r"<[^>]+>", " ", txt))
            return {"ok": True, "tool": "fetch_url", "output": re.sub(r"\s+", " ", txt).strip()[:max_chars]}
        except Exception as e:
            return {"ok": False, "tool": "fetch_url", "error": f"{type(e).__name__}: {str(e)[:200]}"}

    # -- dispatch ------------------------------------------------------------
    def call(self, name: str, args: dict) -> dict:
        fn = {"run_shell": self.run_shell, "run_python": self.run_python, "read_file": self.read_file,
              "write_file": self.write_file, "list_dir": self.list_dir,
              "web_search": self.web_search, "fetch_url": self.fetch_url}.get(name)
        if fn is None:
            return {"ok": False, "tool": name, "error": f"unknown tool '{name}'"}
        try:
            return fn(**args)
        except TypeError as e:
            return {"ok": False, "tool": name, "error": f"bad args: {e}"}

    def schema(self) -> str:
        base = "run_shell(command), run_python(code), read_file(path), write_file(path,content), list_dir(path)"
        if self.allow_web:
            base += ", web_search(query), fetch_url(url)"
        return base


if __name__ == "__main__":
    t = Tools()
    print("schema:", t.schema())
    print("safe   :", t.run_shell("python --version"))
    print("denied :", t.run_shell("git push --force origin main"))
    print("write  :", t.write_file("_tools_selftest.txt", "ok"))
    print("read   :", t.read_file("_tools_selftest.txt"))
