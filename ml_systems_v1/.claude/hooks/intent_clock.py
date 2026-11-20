#!/usr/bin/env python3
"""UserPromptSubmit hook (H3) -- inject the intent-alignment preflight + a fresh clock.

Output on stdout (exit 0) is added to Claude's context before it processes the prompt.
Re-surfaces the spirit>=letter preflight EVERY turn so it cannot decay mid-context, and
stamps a VERIFIED wall-clock reading so elapsed-time is never estimated by feel.

PROPOSED / STAGED. Composes with (appends to) the existing UserPromptSubmit hooks
(autonomous-mode, @browser) -- does not replace them. No emoji (cp1252).
"""
import subprocess
import sys

try:
    now = subprocess.check_output(["date"], text=True).strip()
except Exception:
    now = "(date unavailable -- run `date` before any elapsed claim)"

print(
    "[INTENT PREFLIGHT] Before acting, reconstruct: (1) LITERAL ask (2) SPIRIT/why "
    "(3) DIVERGENCE letter-vs-spirit (4) ALTITUDE: is this the right problem? "
    "(5) EFFECTIVE objective. Spirit >= letter; surface divergences in one line; do not "
    "rigidly literalize a constraint into paralysis nor liberally discard an explicit gate. "
    "Honor RWYB + no-inflation + self-audit + sandbox->review->push. "
    "Wall-clock VERIFIED now: " + now
)
sys.exit(0)
