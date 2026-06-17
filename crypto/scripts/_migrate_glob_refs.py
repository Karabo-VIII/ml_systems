"""Update remaining src/v* glob refs to src/wm/v*."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FILES = [
    "CLAUDE.md",
    ".claude/skills/auditor/SKILL.md",
    ".claude/skills/deep/SKILL.md",
    ".claude/skills/apex/SKILL.md",
    ".claude/agent_state/tracker.py",
    "scripts/audit_vmodel_drift.py",
    "docs/DOUBLE_AUDIT_PROTOCOL.md",
    "memory/research_delegation_protocol.md",
]

REPLACEMENTS = [
    ("src/v*/v*_training/", "src/wm/v*/v*_training/"),
    ("src/v*/`",            "src/wm/v*/`"),
    ("src/v*/ ",            "src/wm/v*/ "),
    ("src/v*/\n",           "src/wm/v*/\n"),
]

for f in FILES:
    p = ROOT / f
    if not p.exists():
        print(f"  SKIP missing: {f}")
        continue
    txt = p.read_text(encoding="utf-8")
    orig = txt
    for old, new in REPLACEMENTS:
        txt = txt.replace(old, new)
    if txt != orig:
        p.write_text(txt, encoding="utf-8")
        print(f"  UPDATED: {f}")
    else:
        print(f"  nochange: {f}")
print("done")
