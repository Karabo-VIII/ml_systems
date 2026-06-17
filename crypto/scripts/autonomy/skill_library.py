#!/usr/bin/env python3
"""scripts/autonomy/skill_library.py -- REUSABLE VALIDATED ASSET registry (Voyager-style skill library).

PURPOSE
-------
The autonomous runner (AUTONOMOUS_RUNNER.md §5) promises a "reusable-asset register" so agents REUSE
tools they have already built instead of re-discovering them.  This is the mechanical implementation.

The registry lives at runs/autonomy/skill_library/INDEX.json.  Every time an agent builds + validates
a reusable artifact (tool / probe / harness / engine / gate / dataset), it calls `register()` once.
Every time a new cycle starts, it calls `digest()` to get a compact, prompt-ready string of the most
relevant assets -- making "reuse-before-build" MECHANICAL, not aspirational.

KINDS: tool | probe | harness | engine | gate | dataset

API (Python)
------------
  register(name, kind, path, entrypoint, signature, summary, tested_on,
           provenance_sha, tags) -> dict  (the entry written)
  search(query, k=10) -> list[dict]
  list_assets(kind=None) -> list[dict]
  digest(n=10, query=None) -> str   (compact, prompt-ready)

CLI
---
  python scripts/autonomy/skill_library.py list [--kind K]
  python scripts/autonomy/skill_library.py search <query>
  python scripts/autonomy/skill_library.py digest [--n N] [--query Q]
  python scripts/autonomy/skill_library.py register --name N --kind K --path P --entrypoint E
      --signature S --summary T --tested-on O --provenance-sha H --tags t1,t2

No emoji (Windows cp1252 safety).  Atomic writes via tmp-rename contract.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = ROOT / "runs" / "autonomy" / "skill_library" / "INDEX.json"

# ---------------------------------------------------------------------------
# Low-level I/O (atomic write so a crash never corrupts the index)
# ---------------------------------------------------------------------------

def _load() -> dict:
    if not INDEX_PATH.exists():
        return {"assets": []}
    try:
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"assets": []}


def _save(data: dict) -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = INDEX_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    # atomic rename (Windows: replace target if it exists)
    try:
        os.replace(str(tmp), str(INDEX_PATH))
    except Exception:
        tmp.rename(INDEX_PATH)


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def register(
    name: str,
    kind: str,
    path: str,
    entrypoint: str,
    signature: str,
    summary: str,
    tested_on: str,
    provenance_sha: str,
    tags: list[str] | None = None,
    added_ts: int | None = None,
) -> dict:
    """Write or update one asset entry in INDEX.json.

    Parameters
    ----------
    name          : unique short identifier, e.g. "candidate_gate"
    kind          : one of {tool, probe, harness, engine, gate, dataset}
    path          : repo-relative path to the file, e.g. "src/strat/candidate_gate.py"
    entrypoint    : callable name, e.g. "evaluate_candidate"
    signature     : brief Python-ish signature, e.g. "(harness, family_n=None) -> dict"
    summary       : 1-2 sentence plain-English description
    tested_on     : what it has been tested on, e.g. "SOL 4h + BTC 1d (RWYB 2026-06-05)"
    provenance_sha: git SHA at the time the asset was validated
    tags          : list of searchable labels, e.g. ["validation", "gate", "strat"]
    added_ts      : Unix timestamp (default: int(time.time()))
    """
    valid_kinds = {"tool", "probe", "harness", "engine", "gate", "dataset"}
    if kind not in valid_kinds:
        raise ValueError(f"kind must be one of {valid_kinds}, got '{kind}'")

    data = _load()
    assets = data.get("assets", [])

    entry = {
        "name": name,
        "kind": kind,
        "path": path,
        "entrypoint": entrypoint,
        "signature": signature,
        "summary": summary,
        "tested_on": tested_on,
        "provenance_sha": provenance_sha,
        "tags": tags or [],
        "added_ts": added_ts if added_ts is not None else int(time.time()),
    }

    # update-in-place if name already exists, otherwise append
    replaced = False
    for i, a in enumerate(assets):
        if a.get("name") == name:
            assets[i] = entry
            replaced = True
            break
    if not replaced:
        assets.append(entry)

    data["assets"] = assets
    _save(data)
    action = "updated" if replaced else "registered"
    print(f"[skill_library] {action}: {name!r} ({kind}) -> {path}")
    return entry


def list_assets(kind: str | None = None) -> list[dict]:
    """Return all assets, optionally filtered by kind."""
    assets = _load().get("assets", [])
    if kind:
        assets = [a for a in assets if a.get("kind") == kind]
    return assets


def search(query: str, k: int = 10) -> list[dict]:
    """Fuzzy-ish keyword search across name, summary, tags, entrypoint.

    Scores by number of query tokens matched (case-insensitive).  Returns top-k.
    """
    tokens = query.lower().split()
    assets = _load().get("assets", [])

    def _score(a: dict) -> int:
        haystack = " ".join([
            a.get("name", ""),
            a.get("summary", ""),
            a.get("entrypoint", ""),
            " ".join(a.get("tags", [])),
            a.get("kind", ""),
        ]).lower()
        return sum(1 for t in tokens if t in haystack)

    scored = [(a, _score(a)) for a in assets]
    scored = [(a, s) for a, s in scored if s > 0]
    scored.sort(key=lambda x: -x[1])
    return [a for a, _ in scored[:k]]


def digest(n: int = 10, query: str | None = None) -> str:
    """Compact, prompt-ready string of the most relevant assets.

    Call this at every cycle start so "reuse-before-build" is mechanical.
    If query is given, returns the top-n matching.  Otherwise returns the n most recently added.
    """
    if query:
        assets = search(query, k=n)
    else:
        all_assets = _load().get("assets", [])
        assets = sorted(all_assets, key=lambda a: a.get("added_ts", 0), reverse=True)[:n]

    if not assets:
        return "(skill_library: empty -- register assets after building them)"

    lines = ["REUSABLE VALIDATED ASSETS (read-forward: reuse before build):"]
    for a in assets:
        tags_str = ", ".join(a.get("tags", [])) or "—"
        lines.append(
            f"  [{a['kind'].upper()}] {a['name']}"
            f"\n    path: {a['path']}  entrypoint: {a['entrypoint']}"
            f"\n    sig: {a['signature']}"
            f"\n    summary: {a['summary']}"
            f"\n    tested_on: {a['tested_on']}"
            f"\n    tags: {tags_str}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_list(args):
    assets = list_assets(kind=getattr(args, "kind", None))
    if not assets:
        print("(no assets registered yet)")
        return
    print(f"=== skill_library: {len(assets)} asset(s) ===")
    for a in assets:
        tags_str = ", ".join(a.get("tags", [])) or "—"
        print(f"  [{a['kind'].upper()}] {a['name']:30s}  {a['path']}")
        print(f"    entrypoint: {a['entrypoint']}  sig: {a['signature']}")
        print(f"    summary: {a['summary']}")
        print(f"    tested_on: {a['tested_on']}")
        print(f"    tags: {tags_str}")
        print()


def _cmd_search(args):
    results = search(args.query)
    if not results:
        print(f"(no matches for '{args.query}')")
        return
    print(f"=== skill_library: {len(results)} match(es) for '{args.query}' ===")
    for a in results:
        print(f"  [{a['kind'].upper()}] {a['name']}  --  {a['summary']}")


def _cmd_digest(args):
    print(digest(n=getattr(args, "n", 10), query=getattr(args, "query", None)))


def _cmd_register(args):
    tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()]
    register(
        name=args.name,
        kind=args.kind,
        path=args.path,
        entrypoint=args.entrypoint,
        signature=args.signature,
        summary=args.summary,
        tested_on=args.tested_on,
        provenance_sha=args.provenance_sha,
        tags=tags,
    )


def main():
    ap = argparse.ArgumentParser(
        prog="skill_library",
        description="Reusable validated asset registry for the autonomous runner.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # list
    lp = sub.add_parser("list", help="List registered assets.")
    lp.add_argument("--kind", default=None, help="Filter by kind (tool/probe/harness/engine/gate/dataset).")

    # search
    sp = sub.add_parser("search", help="Search assets by keyword.")
    sp.add_argument("query", help="Search query (space-separated tokens).")

    # digest
    dp = sub.add_parser("digest", help="Print prompt-ready digest.")
    dp.add_argument("--n", type=int, default=10, help="Max assets to include (default 10).")
    dp.add_argument("--query", default=None, help="Optional search query to filter by.")

    # register
    rp = sub.add_parser("register", help="Register a new or updated asset.")
    rp.add_argument("--name", required=True)
    rp.add_argument("--kind", required=True, choices=["tool", "probe", "harness", "engine", "gate", "dataset"])
    rp.add_argument("--path", required=True)
    rp.add_argument("--entrypoint", required=True)
    rp.add_argument("--signature", required=True)
    rp.add_argument("--summary", required=True)
    rp.add_argument("--tested-on", dest="tested_on", required=True)
    rp.add_argument("--provenance-sha", dest="provenance_sha", required=True)
    rp.add_argument("--tags", default="", help="Comma-separated tags.")

    args = ap.parse_args()
    {
        "list": _cmd_list,
        "search": _cmd_search,
        "digest": _cmd_digest,
        "register": _cmd_register,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
