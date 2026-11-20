"""
Agent State Tracker — Zero-API local state management for expert agents.

Scans the project, computes file hashes, detects changes, generates per-agent
briefings and a comprehensive project snapshot. Runs entirely locally with no
LLM calls. Designed to be run before agent invocations or on a schedule.

Usage:
    python .claude/agent_state/tracker.py                # Full scan + briefings
    python .claude/agent_state/tracker.py --snapshot     # Snapshot only
    python .claude/agent_state/tracker.py --briefing pipeline  # Single agent
    python .claude/agent_state/tracker.py --changes      # Show changes since last scan
"""

import json
import hashlib
import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Set

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATE_DIR = Path(__file__).resolve().parent
BRIEFING_DIR = STATE_DIR / "briefings"
REGISTRY_PATH = STATE_DIR / "file_registry.json"
SNAPSHOT_PATH = STATE_DIR / "project_snapshot.json"
CHANGELOG_PATH = STATE_DIR / "change_log.jsonl"

# File patterns to track, grouped by domain
DOMAIN_FILES = {
    "pipeline": [
        "src/pipeline/*.py",
        "config/*.yaml",
    ],
    "architect": [
        "src/v*/v*/components.py",      # note: glob expanded below
        "src/v*/v*/world_model.py",
        "src/v*/v*/settings.py",
    ],
    "trainer": [
        "src/v*/v*/train_world_model.py",
        "src/v*/v*/settings.py",
        "src/anti_fragile.py",
        "src/wm/v1/v1_training/train_adapter.py",
    ],
    "validator": [
        "src/validation_utils.py",
        "src/v*/v*/validate_world.py",
    ],
    "trader": [
        "src/wm/v4/v4_training/agent.py",
        "src/wm/v4/v4_training/train_agent.py",
        "src/v*/v*/settings.py",
    ],
    "researcher": [
        "*.md",                       # project-root markdown docs
        "src/v*/v*/settings.py",
    ],
    "auditor": [
        "src/**/*.py",                # everything
        "CLAUDE.md",
    ],
    "deep": [
        "src/**/*.py",
        ".claude/skills/*/SKILL.md",
        ".claude/agents/*.md",
    ],
    "meta": [
        ".claude/skills/*/SKILL.md",
        ".claude/agents/*.md",
        "CLAUDE.md",
        "EXPERTS_README.md",
    ],
}

# Directories to always skip
SKIP_DIRS = {".venv", "__pycache__", ".git", "node_modules", ".mypy_cache",
             "models", "logs", "data", "plots", ".claude/agent_state"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def file_hash(path: Path) -> str:
    """SHA256 of file contents."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except (OSError, PermissionError):
        return "unreadable"


def file_mtime(path: Path) -> float:
    """Modification time as Unix timestamp."""
    try:
        return path.stat().st_mtime
    except (OSError, PermissionError):
        return 0.0


def glob_expand(pattern: str) -> List[Path]:
    """Expand a glob pattern relative to PROJECT_ROOT."""
    return sorted(PROJECT_ROOT.glob(pattern))


def is_tracked(path: Path) -> bool:
    """Should this file be tracked?"""
    rel = path.relative_to(PROJECT_ROOT)
    parts = rel.parts
    for skip in SKIP_DIRS:
        if skip in parts:
            return False
    if path.suffix in (".pyc", ".pyo", ".so", ".dll", ".egg-info"):
        return False
    return True


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Registry: file hash tracking
# ---------------------------------------------------------------------------

def load_registry() -> Dict:
    """Load previous file registry."""
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r") as f:
            return json.load(f)
    return {"files": {}, "last_scan": None}


def save_registry(registry: Dict):
    """Save file registry."""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def scan_all_files() -> Dict[str, Dict]:
    """Scan all tracked files, return {relative_path: {hash, mtime, size}}."""
    result = {}
    # Scan src/
    for py_file in PROJECT_ROOT.rglob("*.py"):
        if is_tracked(py_file):
            rel = str(py_file.relative_to(PROJECT_ROOT)).replace("\\", "/")
            result[rel] = {
                "hash": file_hash(py_file),
                "mtime": file_mtime(py_file),
                "size": py_file.stat().st_size,
            }
    # Scan config/
    for yaml_file in (PROJECT_ROOT / "config").glob("*.yaml"):
        if is_tracked(yaml_file):
            rel = str(yaml_file.relative_to(PROJECT_ROOT)).replace("\\", "/")
            result[rel] = {
                "hash": file_hash(yaml_file),
                "mtime": file_mtime(yaml_file),
                "size": yaml_file.stat().st_size,
            }
    # Scan root markdown
    for md_file in PROJECT_ROOT.glob("*.md"):
        if is_tracked(md_file):
            rel = str(md_file.relative_to(PROJECT_ROOT)).replace("\\", "/")
            result[rel] = {
                "hash": file_hash(md_file),
                "mtime": file_mtime(md_file),
                "size": md_file.stat().st_size,
            }
    # Scan skill files
    for skill_file in (PROJECT_ROOT / ".claude" / "skills").rglob("SKILL.md"):
        rel = str(skill_file.relative_to(PROJECT_ROOT)).replace("\\", "/")
        result[rel] = {
            "hash": file_hash(skill_file),
            "mtime": file_mtime(skill_file),
            "size": skill_file.stat().st_size,
        }
    # Scan agent definition files
    agents_dir = PROJECT_ROOT / ".claude" / "agents"
    if agents_dir.exists():
        for agent_file in agents_dir.glob("*.md"):
            rel = str(agent_file.relative_to(PROJECT_ROOT)).replace("\\", "/")
            result[rel] = {
                "hash": file_hash(agent_file),
                "mtime": file_mtime(agent_file),
                "size": agent_file.stat().st_size,
            }
    return result


# ---------------------------------------------------------------------------
# Change detection
# ---------------------------------------------------------------------------

def detect_changes(old_files: Dict, new_files: Dict) -> Dict:
    """Compare two file registries, return categorized changes."""
    old_keys = set(old_files.keys())
    new_keys = set(new_files.keys())

    added = new_keys - old_keys
    removed = old_keys - new_keys
    modified = set()
    unchanged = set()

    for key in old_keys & new_keys:
        if old_files[key]["hash"] != new_files[key]["hash"]:
            modified.add(key)
        else:
            unchanged.add(key)

    return {
        "added": sorted(added),
        "removed": sorted(removed),
        "modified": sorted(modified),
        "unchanged": sorted(unchanged),
        "total_changed": len(added) + len(removed) + len(modified),
    }


def route_changes_to_agents(changes: Dict) -> Dict[str, List[str]]:
    """Map changed files to which agents care about them."""
    all_changed = set(changes["added"] + changes["removed"] + changes["modified"])
    agent_changes: Dict[str, List[str]] = {agent: [] for agent in DOMAIN_FILES}

    for agent, patterns in DOMAIN_FILES.items():
        for pattern in patterns:
            matched_files = glob_expand(pattern)
            for mf in matched_files:
                rel = str(mf.relative_to(PROJECT_ROOT)).replace("\\", "/")
                if rel in all_changed:
                    agent_changes[agent].append(rel)
        # Deduplicate
        agent_changes[agent] = sorted(set(agent_changes[agent]))

    return agent_changes


# ---------------------------------------------------------------------------
# Project snapshot
# ---------------------------------------------------------------------------

def build_snapshot(files: Dict) -> Dict:
    """Build comprehensive project snapshot."""
    snapshot = {
        "timestamp": now_iso(),
        "file_count": len(files),
    }

    # --- Model checkpoints ---
    checkpoints = {}
    for v in range(1, 10):
        model_dir = PROJECT_ROOT / "models" / f"v{v}"
        if model_dir.exists():
            ckpts = sorted(model_dir.glob("*.pt"))
            if ckpts:
                best_ema = [c for c in ckpts if "best_ema" in c.name]
                latest = ckpts[-1]
                checkpoints[f"v{v}"] = {
                    "count": len(ckpts),
                    "latest": latest.name,
                    "latest_size_mb": round(latest.stat().st_size / 1e6, 1),
                    "latest_date": datetime.fromtimestamp(
                        latest.stat().st_mtime, tz=timezone.utc
                    ).strftime("%Y-%m-%d %H:%M"),
                    "has_best_ema": len(best_ema) > 0,
                }
    snapshot["checkpoints"] = checkpoints

    # --- Dataset status ---
    datasets = {}
    data_dir = PROJECT_ROOT / "data" / "processed"
    if data_dir.exists():
        for pq in sorted(data_dir.glob("*_v50_chimera.parquet")):
            symbol = pq.name.split("_")[0].upper()
            datasets[symbol] = {
                "file": pq.name,
                "size_mb": round(pq.stat().st_size / 1e6, 1),
                "date": datetime.fromtimestamp(
                    pq.stat().st_mtime, tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M"),
            }
    snapshot["datasets"] = datasets

    # --- Source code stats ---
    version_stats = {}
    for v in range(1, 10):
        v_dir = PROJECT_ROOT / "src" / f"v{v}_training"
        if v_dir.exists():
            py_files = list(v_dir.glob("*.py"))
            total_lines = 0
            file_list = []
            for pf in py_files:
                if pf.name.startswith("__"):
                    continue
                try:
                    lines = len(pf.read_text(encoding="utf-8", errors="replace").splitlines())
                except Exception:
                    lines = 0
                total_lines += lines
                file_list.append(f"{pf.name} ({lines}L)")
            version_stats[f"v{v}"] = {
                "files": file_list,
                "total_lines": total_lines,
            }
    snapshot["source_stats"] = version_stats

    # --- Shared module stats ---
    shared = {}
    for name in ["anti_fragile.py", "validation_utils.py", "log_utils.py", "preflight_checks.py"]:
        p = PROJECT_ROOT / "src" / name
        if p.exists():
            try:
                lines = len(p.read_text(encoding="utf-8", errors="replace").splitlines())
            except Exception:
                lines = 0
            shared[name] = {"lines": lines, "size_kb": round(p.stat().st_size / 1024, 1)}
    snapshot["shared_modules"] = shared

    # --- Training logs ---
    logs = {}
    for v in range(1, 10):
        log_dir = PROJECT_ROOT / "logs" / f"v{v}"
        if log_dir.exists():
            log_files = sorted(log_dir.glob("*.log")) + sorted(log_dir.glob("*.txt"))
            if log_files:
                latest = log_files[-1]
                logs[f"v{v}"] = {
                    "count": len(log_files),
                    "latest": latest.name,
                    "latest_date": datetime.fromtimestamp(
                        latest.stat().st_mtime, tz=timezone.utc
                    ).strftime("%Y-%m-%d %H:%M"),
                }
    snapshot["training_logs"] = logs

    return snapshot


# ---------------------------------------------------------------------------
# Agent briefings
# ---------------------------------------------------------------------------

def generate_briefing(agent: str, changes: Dict, agent_changes: Dict[str, List[str]],
                      snapshot: Dict, agent_state: Optional[Dict]) -> str:
    """Generate a markdown briefing for a specific agent."""
    my_changes = agent_changes.get(agent, [])
    now = now_iso()

    lines = [
        f"# {agent.title()} Agent Briefing",
        f"Generated: {now}",
        "",
    ]

    # --- Change summary ---
    if changes["total_changed"] == 0:
        lines.append("## Changes Since Last Scan")
        lines.append("No files changed since last scan.")
    else:
        lines.append(f"## Changes Since Last Scan ({changes['total_changed']} total)")
        if my_changes:
            lines.append(f"\n**Files in YOUR domain that changed ({len(my_changes)}):**")
            for f in my_changes:
                change_type = "NEW" if f in changes["added"] else "MODIFIED" if f in changes["modified"] else "REMOVED"
                lines.append(f"- `{f}` [{change_type}]")
        else:
            lines.append("\nNo files in your domain changed.")

        # Other agents' changes (brief)
        other_changes = {a: ch for a, ch in agent_changes.items() if a != agent and ch}
        if other_changes:
            lines.append("\n**Changes in other domains:**")
            for a, ch in other_changes.items():
                lines.append(f"- {a}: {len(ch)} file(s) changed")
    lines.append("")

    # --- Project snapshot summary (agent-relevant) ---
    lines.append("## Project Status")

    # Checkpoints
    if snapshot.get("checkpoints"):
        lines.append("\n**Model Checkpoints:**")
        for v, info in snapshot["checkpoints"].items():
            ema_tag = " [has EMA]" if info.get("has_best_ema") else ""
            lines.append(
                f"- {v}: {info['count']} checkpoints, latest={info['latest']} "
                f"({info['latest_size_mb']}MB, {info['latest_date']}){ema_tag}"
            )
    else:
        lines.append("\n**Model Checkpoints:** None found")

    # Datasets
    if snapshot.get("datasets"):
        lines.append(f"\n**Datasets:** {len(snapshot['datasets'])} assets")
        for sym, info in snapshot["datasets"].items():
            lines.append(f"- {sym}: {info['size_mb']}MB ({info['date']})")
    else:
        lines.append("\n**Datasets:** None found")

    # Agent-specific sections
    if agent == "trainer" and snapshot.get("training_logs"):
        lines.append("\n**Training Logs:**")
        for v, info in snapshot["training_logs"].items():
            lines.append(f"- {v}: {info['count']} logs, latest={info['latest']} ({info['latest_date']})")

    if agent in ("architect", "deep") and snapshot.get("source_stats"):
        lines.append("\n**Source Code Stats:**")
        for v, info in snapshot["source_stats"].items():
            lines.append(f"- {v}: {info['total_lines']}L across {len(info['files'])} files")

    if agent in ("auditor", "meta"):
        lines.append("\n**Shared Modules:**")
        for name, info in snapshot.get("shared_modules", {}).items():
            lines.append(f"- {name}: {info['lines']}L ({info['size_kb']}KB)")

    lines.append("")

    # --- Previous findings (if state exists) ---
    if agent_state and agent_state.get("findings"):
        findings = agent_state["findings"]
        open_count = sum(1 for f in findings if f.get("status") == "open")
        resolved_count = sum(1 for f in findings if f.get("status") == "resolved")
        lines.append(f"## Previous Findings ({open_count} open, {resolved_count} resolved)")
        for finding in findings:
            if finding.get("status") == "open":
                lines.append(
                    f"- [{finding.get('severity', '?')}] {finding.get('title', 'Untitled')} "
                    f"(found {finding.get('date', '?')})"
                )
                if finding.get("file"):
                    lines.append(f"  File: `{finding['file']}`")
        lines.append("")
    else:
        lines.append("## Previous Findings")
        lines.append("No previous findings recorded. This may be the first run.")
        lines.append("")

    # --- Last review info ---
    if agent_state and agent_state.get("last_review"):
        lr = agent_state["last_review"]
        lines.append(f"## Last Review")
        lines.append(f"- Date: {lr.get('date', '?')}")
        lines.append(f"- Scope: {lr.get('scope', '?')}")
        lines.append(f"- Files reviewed: {lr.get('files_reviewed', '?')}")
        lines.append("")

    return "\n".join(lines)


def load_agent_state(agent: str) -> Optional[Dict]:
    """Load persisted state for an agent."""
    path = STATE_DIR / f"{agent}_state.json"
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def save_agent_state(agent: str, state: Dict):
    """Save agent state."""
    path = STATE_DIR / f"{agent}_state.json"
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def init_agent_state(agent: str) -> Dict:
    """Create initial empty state for an agent."""
    return {
        "agent": agent,
        "created": now_iso(),
        "last_review": None,
        "findings": [],
        "reviewed_files": {},
        "notes": [],
    }


# ---------------------------------------------------------------------------
# Change log
# ---------------------------------------------------------------------------

def append_changelog(changes: Dict, agent_changes: Dict):
    """Append a change entry to the changelog."""
    if changes["total_changed"] == 0:
        return
    entry = {
        "timestamp": now_iso(),
        "added": changes["added"],
        "removed": changes["removed"],
        "modified": changes["modified"],
        "affected_agents": {a: ch for a, ch in agent_changes.items() if ch},
    }
    with open(CHANGELOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Agent State Tracker")
    parser.add_argument("--snapshot", action="store_true", help="Generate snapshot only")
    parser.add_argument("--briefing", type=str, help="Generate briefing for specific agent")
    parser.add_argument("--changes", action="store_true", help="Show changes since last scan")
    parser.add_argument("--init", action="store_true", help="Initialize all agent states")
    args = parser.parse_args()

    BRIEFING_DIR.mkdir(parents=True, exist_ok=True)

    # Load previous registry
    old_registry = load_registry()
    old_files = old_registry.get("files", {})

    # Scan current files
    print(f"[SCAN] Scanning project files...")
    new_files = scan_all_files()
    print(f"[SCAN] Found {len(new_files)} tracked files")

    # Detect changes
    changes = detect_changes(old_files, new_files)
    agent_changes = route_changes_to_agents(changes)

    if args.changes:
        print(f"\n=== CHANGES SINCE LAST SCAN ===")
        print(f"Added:     {len(changes['added'])}")
        print(f"Modified:  {len(changes['modified'])}")
        print(f"Removed:   {len(changes['removed'])}")
        print(f"Unchanged: {len(changes['unchanged'])}")
        if changes["total_changed"] > 0:
            print(f"\nChanged files:")
            for f in changes["added"]:
                print(f"  + {f}")
            for f in changes["modified"]:
                print(f"  ~ {f}")
            for f in changes["removed"]:
                print(f"  - {f}")
            print(f"\nAffected agents:")
            for agent, files in agent_changes.items():
                if files:
                    print(f"  {agent}: {len(files)} file(s)")
        return

    # Build snapshot
    print(f"[SNAP] Building project snapshot...")
    snapshot = build_snapshot(new_files)
    with open(SNAPSHOT_PATH, "w") as f:
        json.dump(snapshot, f, indent=2)
    print(f"[SNAP] Snapshot saved: {len(snapshot.get('checkpoints', {}))} model versions, "
          f"{len(snapshot.get('datasets', {}))} datasets")

    if args.snapshot:
        print(json.dumps(snapshot, indent=2))
        return

    # Initialize agent states if needed
    if args.init:
        for agent in DOMAIN_FILES:
            state = init_agent_state(agent)
            save_agent_state(agent, state)
            print(f"[INIT] Initialized {agent} state")
        print("[INIT] All agent states initialized")

    # Generate briefings
    agents_to_brief = [args.briefing] if args.briefing else list(DOMAIN_FILES.keys())
    for agent in agents_to_brief:
        if agent not in DOMAIN_FILES:
            print(f"[WARN] Unknown agent: {agent}")
            continue
        agent_state = load_agent_state(agent)
        if agent_state is None:
            agent_state = init_agent_state(agent)
            save_agent_state(agent, agent_state)
        briefing = generate_briefing(agent, changes, agent_changes, snapshot, agent_state)
        briefing_path = BRIEFING_DIR / f"{agent}.md"
        with open(briefing_path, "w", encoding="utf-8") as f:
            f.write(briefing)
        change_count = len(agent_changes.get(agent, []))
        print(f"[BRIEF] {agent}: {change_count} changed file(s) in domain")

    # Update registry
    new_registry = {
        "files": new_files,
        "last_scan": now_iso(),
        "scan_count": old_registry.get("scan_count", 0) + 1,
    }
    save_registry(new_registry)

    # Append to changelog
    append_changelog(changes, agent_changes)

    print(f"\n[DONE] Scan #{new_registry['scan_count']} complete. "
          f"{changes['total_changed']} changes detected.")


if __name__ == "__main__":
    main()
