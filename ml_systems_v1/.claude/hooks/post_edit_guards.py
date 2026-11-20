#!/usr/bin/env python3
"""PostToolUse hook (H5 + H6) -- non-blocking guards fired right after an Edit/Write.

H5 emoji-guard: flags emoji in a .py edit (Windows cp1252 crash invariant).
H6 RWYB nudge: reminds to run-what-you-build when strategy/training/cost code changes.

Non-blocking: prints to stdout (exit 0) as a nudge; never blocks the edit. PROPOSED / STAGED.
The emoji ranges below are written as \\U escapes (ASCII), so this file is itself cp1252-safe.
"""
import json
import re
import sys

STRAT_HINT = re.compile(r"(strat|train|cost|wealth_bot|sim|backtest|sizing|harness)", re.I)
EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\U00002190-\U000021FF\U00002B00-\U00002BFF]"
)

try:
    sys.stdin.reconfigure(encoding="utf-8")  # Windows cp1252 default would drop emoji bytes
except Exception:
    pass
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)

ti = data.get("tool_input", {}) or {}
fp = ti.get("file_path", "") or ""
content = (ti.get("content") or ti.get("new_string") or "")

msgs = []
if fp.endswith(".py") and EMOJI.search(content):
    msgs.append(
        "EMOJI-GUARD: emoji detected in a .py edit -- remove from any print/log (Windows cp1252 crashes)."
    )
if fp.endswith(".py") and STRAT_HINT.search(fp):
    msgs.append(
        "RWYB: strategy/training/cost code changed -- run it against real data and document the "
        "command+result before commit (Layer-1 invariant)."
    )

if msgs:
    print(" ".join(msgs))
sys.exit(0)
