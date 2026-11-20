"""doc_audit_crawler.py -- audit project documentation for sprawl.

User mandate: "check whole project documentation block by block ... overlaps,
dups, etc". This crawler detects:

  1. TOPIC-CLUSTER  : multiple docs sharing a name prefix (likely overlapping)
  2. DUP-CONTENT    : exact content duplicates (sha1 hash match)
  3. NEAR-DUP       : title + first-paragraph hash match (likely copy-paste)
  4. ORPHAN         : doc not linked from any .md or .py file
  5. STALE          : mtime older than --stale-days threshold (default 60d)
  6. SUPERSEDED     : doc explicitly marked superseded OR a sibling references
                       "supersedes <this>"
  7. NAMING-DRIFT   : inconsistent date suffixes (mixed _2026_05_16 vs
                       _2026-05-16 vs no suffix)

OUTPUT
------
runs/audit/doc_audit_<DATE>.md -- per-category findings + recommended
consolidation queue.

NON-INVASIVE
============
Read-only audit. Never deletes / renames / moves docs. Surfaces findings
for human review; consolidation actions are explicit user decisions.
"""
from __future__ import annotations

__contract__ = {
    "kind": "doc_audit_crawler",
    "owner": "audit/docs",
    "outputs": ["runs/audit/doc_audit_<DATE>.md"],
    "invariants": [
        "read-only; never deletes / renames / moves",
        "complements existing 6 pipeline crawlers (codebase + data layers)",
    ],
}

import argparse
import datetime as dt
import hashlib
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = PROJECT_ROOT / "docs"
OUT_DIR = PROJECT_ROOT / "runs" / "audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def list_docs() -> list[Path]:
    """All non-archive markdown docs (root + docs/)."""
    out: list[Path] = []
    # Root-level docs
    for fp in PROJECT_ROOT.glob("*.md"):
        out.append(fp)
    # docs/ tree but skip archive directories
    for fp in DOCS_DIR.rglob("*.md"):
        parts = fp.relative_to(DOCS_DIR).parts
        if parts and any(p.startswith("archive") for p in parts):
            continue
        out.append(fp)
    return sorted(out)


# ============================================================================
# Axis 1: topic-cluster (name-prefix overlap)
# ============================================================================

def audit_topic_clusters(docs: list[Path], min_cluster_size: int = 3) -> list[dict]:
    """Group docs by name prefix (first 1-2 alphanumeric tokens). Flag
    clusters >= min_cluster_size as likely-overlapping."""
    # Extract a normalized cluster-key per doc
    by_cluster: dict[str, list[Path]] = defaultdict(list)
    for fp in docs:
        stem = fp.stem
        # Strip trailing date suffix (_YYYY_MM_DD or _YYYY-MM-DD)
        s = re.sub(r"_\d{4}[_-]\d{2}[_-]\d{2}.*$", "", stem)
        # First 1-2 uppercase tokens form the cluster key
        tokens = re.split(r"[_\-\s]+", s)
        if len(tokens) >= 2:
            key = "_".join(tokens[:2])
        else:
            key = tokens[0] if tokens else stem
        by_cluster[key].append(fp)
    findings: list[dict] = []
    for key, members in by_cluster.items():
        if len(members) >= min_cluster_size:
            findings.append({
                "category": "topic-cluster",
                "cluster_key": key,
                "n_docs": len(members),
                "members": sorted([str(p.relative_to(PROJECT_ROOT))
                                      for p in members]),
                "fix": f"review {len(members)} docs sharing '{key}' prefix; "
                          f"consolidate or archive obsolete ones",
            })
    return findings


# ============================================================================
# Axis 2: dup-content (exact sha1 match)
# ============================================================================

def audit_dup_content(docs: list[Path]) -> list[dict]:
    by_hash: dict[str, list[Path]] = defaultdict(list)
    for fp in docs:
        try:
            data = fp.read_bytes()
        except Exception:
            continue
        if len(data) < 100:
            continue
        h = hashlib.sha1(data).hexdigest()
        by_hash[h].append(fp)
    findings: list[dict] = []
    for h, members in by_hash.items():
        if len(members) >= 2:
            findings.append({
                "category": "dup-content",
                "sha1": h[:12],
                "n_copies": len(members),
                "members": sorted([str(p.relative_to(PROJECT_ROOT))
                                      for p in members]),
                "fix": "DELETE all but one (exact byte-for-byte duplicates)",
            })
    return findings


# ============================================================================
# Axis 3: near-dup (first-paragraph hash)
# ============================================================================

def audit_near_dup(docs: list[Path]) -> list[dict]:
    by_intro: dict[str, list[Path]] = defaultdict(list)
    for fp in docs:
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        # Normalize: lowercase, strip markdown, take first 400 non-WS chars
        normalized = re.sub(r"[#\*\[\]\(\)`>\-_\s]+", " ",
                              text[:2000].lower()).strip()
        if len(normalized) < 200:
            continue
        intro_hash = hashlib.sha1(normalized[:400].encode()).hexdigest()
        by_intro[intro_hash].append(fp)
    findings: list[dict] = []
    for h, members in by_intro.items():
        if len(members) >= 2:
            findings.append({
                "category": "near-dup",
                "intro_hash": h[:12],
                "n_copies": len(members),
                "members": sorted([str(p.relative_to(PROJECT_ROOT))
                                      for p in members]),
                "fix": "manually compare first-paragraph similarity; "
                          "consolidate or archive obsolete copy",
            })
    return findings


# ============================================================================
# Axis 4: orphan (no inbound link)
# ============================================================================

def audit_orphans(docs: list[Path]) -> list[dict]:
    """Doc is orphan if no other .md or .py file mentions its filename."""
    # Build a corpus of all .md + .py text (no docs in archive)
    corpus = []
    for fp in PROJECT_ROOT.rglob("*.md"):
        if any(p.startswith("archive") for p in fp.relative_to(PROJECT_ROOT).parts):
            continue
        try:
            corpus.append(fp.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
    for fp in (PROJECT_ROOT / "src").rglob("*.py"):
        try:
            corpus.append(fp.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
    full_corpus = "\n".join(corpus)
    findings: list[dict] = []
    for fp in docs:
        name = fp.name
        # Count references EXCLUDING self
        refs = full_corpus.count(name) - 1   # -1 for self-reference
        # Also count by stem (e.g. for filename without .md)
        stem_refs = full_corpus.count(fp.stem) - 1
        total_refs = max(refs, stem_refs)
        if total_refs <= 0:
            findings.append({
                "category": "orphan",
                "doc": str(fp.relative_to(PROJECT_ROOT)),
                "name_refs": refs, "stem_refs": stem_refs,
                "fix": "doc never referenced; archive OR link from STATE.md "
                          "/ CLAUDE.md if still relevant",
            })
    return findings


# ============================================================================
# Axis 5: stale (mtime > N days)
# ============================================================================

def audit_stale(docs: list[Path], stale_days: int = 60) -> list[dict]:
    cutoff = dt.datetime.now().timestamp() - stale_days * 86400
    findings: list[dict] = []
    for fp in docs:
        try:
            mtime = fp.stat().st_mtime
        except OSError:
            continue
        if mtime < cutoff:
            age_days = (dt.datetime.now().timestamp() - mtime) / 86400
            findings.append({
                "category": "stale",
                "doc": str(fp.relative_to(PROJECT_ROOT)),
                "age_days": round(age_days, 1),
                "threshold_days": stale_days,
                "fix": f"doc unchanged {age_days:.0f}d; review for "
                          f"archive or refresh",
            })
    return findings


# ============================================================================
# Axis 6: superseded (explicit marker)
# ============================================================================

def audit_superseded(docs: list[Path]) -> list[dict]:
    SUPERSEDE_PATTERNS = [
        r"supersedes?\s+\[?`?(\S+\.md)",
        r"superseded\s+by\s+\[?`?(\S+\.md)",
        r"DEPRECATED",
        r"@deprecated",
    ]
    findings: list[dict] = []
    for fp in docs:
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        # First 1000 chars are usually where supersede markers live
        head = text[:1500]
        for pat in SUPERSEDE_PATTERNS:
            m = re.search(pat, head, re.IGNORECASE)
            if m:
                target = m.group(1) if m.groups() else "(marked deprecated)"
                findings.append({
                    "category": "superseded",
                    "doc": str(fp.relative_to(PROJECT_ROOT)),
                    "pattern": pat,
                    "target": target,
                    "fix": "consider archiving this doc; superseded marker present",
                })
                break
    return findings


# ============================================================================
# Axis 7: naming drift (mixed date-suffix styles)
# ============================================================================

def audit_naming_drift(docs: list[Path]) -> list[dict]:
    # Two date-suffix conventions in use:
    #   _2026_05_16  (underscores)
    #   _2026-05-16  (dashes)
    underscore = []
    dashed = []
    no_suffix = []
    for fp in docs:
        stem = fp.stem
        if re.search(r"_\d{4}_\d{2}_\d{2}", stem):
            underscore.append(fp)
        elif re.search(r"_\d{4}-\d{2}-\d{2}", stem):
            dashed.append(fp)
        elif re.search(r"\d{4}", stem):
            pass    # has year but no date suffix; ambiguous
        else:
            no_suffix.append(fp)
    findings: list[dict] = []
    if underscore and dashed:
        findings.append({
            "category": "naming-drift",
            "underscore_count": len(underscore),
            "dashed_count": len(dashed),
            "underscore_sample": sorted([p.name for p in underscore])[:5],
            "dashed_sample": sorted([p.name for p in dashed])[:5],
            "fix": f"standardize on one convention; project majority is "
                      f"{'underscore' if len(underscore) > len(dashed) else 'dashed'}",
        })
    return findings


# ============================================================================
# Reporter
# ============================================================================

def write_report(all_findings: list[dict]) -> Path:
    today = dt.date.today().isoformat()
    out = OUT_DIR / f"doc_audit_{today}.md"
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for f in all_findings:
        by_cat[f["category"]].append(f)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(f"# Doc Audit -- {today}\n\n")
        fh.write(f"Total findings: {len(all_findings)}\n\n")
        fh.write(f"## Summary by category\n\n")
        for cat, lst in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
            fh.write(f"- **{cat}**: {len(lst)}\n")
        fh.write("\n")
        for cat, lst in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
            fh.write(f"## {cat} ({len(lst)})\n\n")
            shown = lst[:50] if cat in ("orphan", "stale") else lst
            for f in shown:
                fh.write("- finding:\n")
                for k, v in f.items():
                    if k == "category":
                        continue
                    if isinstance(v, list) and len(v) > 8:
                        fh.write(f"  - {k}: [first 8 of {len(v)}]\n")
                        for item in v[:8]:
                            fh.write(f"    - {item}\n")
                    else:
                        fh.write(f"  - {k}: {v}\n")
                fh.write("\n")
            if len(lst) > len(shown):
                fh.write(f"... and {len(lst) - len(shown)} more.\n\n")
    return out


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stale-days", type=int, default=60)
    ap.add_argument("--min-cluster-size", type=int, default=3)
    args = ap.parse_args()

    docs = list_docs()
    print(f"[doc-audit] {len(docs)} non-archive docs in scope")
    findings: list[dict] = []
    print("  axis 1: topic-cluster ...", flush=True)
    findings += audit_topic_clusters(docs, args.min_cluster_size)
    print("  axis 2: dup-content (sha1) ...", flush=True)
    findings += audit_dup_content(docs)
    print("  axis 3: near-dup (intro hash) ...", flush=True)
    findings += audit_near_dup(docs)
    print("  axis 4: orphan ...", flush=True)
    findings += audit_orphans(docs)
    print("  axis 5: stale ...", flush=True)
    findings += audit_stale(docs, args.stale_days)
    print("  axis 6: superseded ...", flush=True)
    findings += audit_superseded(docs)
    print("  axis 7: naming-drift ...", flush=True)
    findings += audit_naming_drift(docs)

    out = write_report(findings)
    print(f"[doc-audit] {len(findings)} findings -> {out}")
    by_cat: dict[str, int] = defaultdict(int)
    for f in findings:
        by_cat[f["category"]] += 1
    for cat, n in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:<22s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
