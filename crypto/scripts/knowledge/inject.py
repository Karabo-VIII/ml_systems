#!/usr/bin/env python3
"""inject.py -- PILLAR 3 packet builder. Produces the KNOWLEDGE PACKET an orchestrator prepends to every dispatched
subagent so a worker is HANDED the firewall (the HARD dead-list digest + the CASH THEOREM + the binding cognition
rules + the intent surface) regardless of how its task is phrased.

The critic's #1 fix, applied: inject the WHOLE HARD dead-list digest UNCONDITIONALLY. Keyword->D-ID matching MISSES
(a 'trade a slice on SOL' task never greps to D40/D48/D55/D75 or the CASH THEOREM, whose rows say 'x-asset',
'universe', '100+ assets', never 'u100'/'SOL'). Always-handing-the-firewall is robust to vocabulary; grep is not.

Usage (orchestrator prepends the output to the Agent/Task prompt):
  python crypto/scripts/knowledge/inject.py
  import inject; packet = inject.build_packet()
No emoji (Windows cp1252).
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CRYPTO = os.path.dirname(os.path.dirname(HERE))
DEAD_LIST = os.path.join(CRYPTO, "docs", "MARKET_FRAMEWORK", "01_DEAD_LIST.md")
COGNITION = os.path.join(CRYPTO, "docs", "COGNITION_RULES.md")
CHARTER = os.path.join(CRYPTO, "CHARTER.md")


def _read(p):
    try:
        with open(p, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except Exception:
        return ""


def hard_dead_digest():
    """One compact line per HARD-scoped killed vein: 'Dxx: <theory> -- <key number>'. Parses the register rows."""
    out = []
    for ln in _read(DEAD_LIST).splitlines():
        if not ln.startswith("| D"):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) < 7:
            continue
        did, theory, _claim, _how, keynum, scope, _src = cells[:7]
        if "HARD" in scope.upper():
            out.append("%s: %s -- %s" % (did, theory, keynum))
    return out


def build_packet(task_text="", domain=""):
    rules = _read(COGNITION).split("## The honest fallback")[0].strip()
    hard = hard_dead_digest()
    ch = _read(CHARTER)
    m = re.search(r"## THE BINDING SURFACE.*?(?=\n## )", ch, re.S)
    charter_surface = m.group(0).strip() if m else "(see crypto/CHARTER.md)"
    parts = [
        "=== KNOWLEDGE PACKET (injected; read BEFORE acting) ===",
        "You are doing crypto research under a binding operating model. Honor the intent; do NOT collapse the "
        "breadth surface (e.g. u100 -> BTC); a null is honest ONLY if the FULL surface was attempted (>=20 u100 "
        "assets AND the window AND existing move-catch infra), proven by run-output artifacts.",
        "",
        "--- BINDING COGNITION RULES ---",
        rules,
        "",
        "--- INTENT SURFACE (crypto/CHARTER.md) ---",
        charter_surface,
        "",
        "--- HARD DEAD VEINS (NEVER re-mine; cite the D-ID; %d entries) ---" % len(hard),
        "\n".join(hard),
        "",
        "Full register + SCOPED veins + recipes: crypto/docs/MARKET_FRAMEWORK/01_DEAD_LIST.md ; "
        "map: crypto/KNOWLEDGE_MAP.md ; rules: crypto/docs/COGNITION_RULES.md",
        "=== END KNOWLEDGE PACKET ===",
    ]
    return "\n".join(parts)


def main(argv):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # packet pulls Unicode from source docs; CLI must not cp1252-crash
    except Exception:
        pass
    domain = argv[0] if argv else ""
    sys.stdout.write(build_packet(domain=domain) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
