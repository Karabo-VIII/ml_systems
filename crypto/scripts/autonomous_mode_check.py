"""UserPromptSubmit hook: inject autonomous-mode status into context EVERY turn.

Reflects the SAME two arming paths the Stop hook (.claude/hooks/autonomy_loop.py) uses, so the banner can never
say OFF while the loop is actually ON:
  A. .claude/autonomous_mode.json  (autonomous=true AND now < envelope_end)  -> TIMED loop
  B. runs/autonomy/AUTONOMY_ON     (the frontier switch file exists)         -> frontier-driven loop
Cheap, never blocks (always exits 0; any error -> treated as that source OFF).
"""
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # crypto/scripts/ -> crypto sub-project root
FLAG = ROOT.parent / ".claude" / "autonomous_mode.json"  # .claude is SHARED at the parent root after the 3-way split
SWITCH = ROOT / "runs" / "autonomy" / "AUTONOMY_ON"      # runs/ is crypto-owned

# UNSKIPPABLE 60s-WATCHER GATE (user mandate 2026-06-06: "no skipping that gate"). Every user turn, if autonomous mode
# is armed and the watcher is down, relaunch it fully-detached. No instance can be in autonomous mode without it.
try:
    sys.path.insert(0, str(ROOT / "scripts" / "autonomy"))
    from ensure_watcher import ensure as _ensure_watcher
    _ensure_watcher()
except Exception:
    pass

envelope_on = False
end = None
mandate = ""
def _parse_envelope(end):
    """Tolerant parse of envelope_end: plain '%Y-%m-%d %H:%M:%S' OR ISO (T + tz). None if unparseable.
    FIX 2026-06-06: an ISO timestamp used to raise -> read fell to except -> banner SILENTLY said OFF (W3 bug)."""
    if not end:
        return None
    try:
        return datetime.datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        pass
    try:
        return datetime.datetime.fromisoformat(end).replace(tzinfo=None)
    except (ValueError, TypeError):
        return None


try:
    d = json.loads(FLAG.read_text(encoding="utf-8"))
    if d.get("autonomous"):
        end = d.get("envelope_end")
        end_dt = _parse_envelope(end)
        if end is None or end_dt is None or datetime.datetime.now() < end_dt:
            envelope_on = True
            mandate = d.get("mandate", "")
except Exception:
    pass  # unreadable/missing flag -> envelope OFF; fall through to the frontier switch

switch_on = SWITCH.exists()

if envelope_on:
    msg = (
        f"AUTONOMOUS MODE: ON (until {end}). You are the OVERSEER (stand-in for the user, Tier-0): own "
        f"objective-fulfillment -- DISPATCH execution + JUDGE results; don't self-execute. Follow "
        f".claude/skills/_common/OVERSEER.md + AUTONOMOUS_RUNNER.md; work the EV-ranked frontier. YOU MAKE THE "
        f"CALLS (Claude manages): COMMIT changes (git is the revert net), do NOT park/defer for anything git can "
        f"revert; escalate ONLY for a genuinely irreversible real-world action. FIX weaknesses as you find them "
        f"(correct-as-you-go). re-`date` before any elapsed claim; a SUMMARY is NOT a stop -- run until the "
        f"objective is verified SOLVED or the window closes. Mandate: {mandate}"
    )
elif switch_on:
    msg = (
        "AUTONOMOUS MODE: ON (frontier loop via runs/autonomy/AUTONOMY_ON). You are the OVERSEER "
        "(.claude/skills/_common/OVERSEER.md): work the EV-ranked frontier at runs/autonomy/frontier.json; FIX "
        "weaknesses as you find them (correct-as-you-go, git-revertible); COMMIT, don't park; re-`date` before "
        "elapsed claims. A SUMMARY is NOT a stop -- the Stop hook continues you until the frontier is empty/"
        "below-floor or the objective is verified SOLVED. Disarm: rm runs/autonomy/AUTONOMY_ON."
    )
else:
    msg = "AUTONOMOUS MODE: OFF"

print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": msg}}))
sys.exit(0)
