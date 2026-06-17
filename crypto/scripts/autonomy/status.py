#!/usr/bin/env python3
"""autonomy status -- the ONE read-only view over the whole control state (harmonization: 'one place to look').

The control state is deliberately spread across several files (each an authority for its concern, per
docs/SYSTEM_TOPOLOGY.md S4). Rather than merge them into one risky state file, this is a read-only AGGREGATOR --
it prints arming + permissions + frontier + gates + reproducibility + the weak-point register in one view. It
adds NO control surface (it only reads). Run: python scripts/autonomy/status.py
No emoji (Windows cp1252).
"""
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(rel):
    try:
        return json.loads((ROOT / rel).read_text(encoding="utf-8"))
    except Exception:
        return None


def main():
    now = datetime.datetime.now()
    print("=" * 70)
    print(f"AUTONOMY STATUS  @ {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # --- ARMING (authority: .claude/autonomous_mode.json) ---
    am = _load(".claude/autonomous_mode.json") or {}
    on = bool(am.get("autonomous"))
    end = am.get("envelope_end")
    mode = "TIMED" if (on and end) else ("UNTIMED(run-until-frontier-empty)" if on else "-")
    left = ""
    if on and end:
        try:
            d = datetime.datetime.strptime(end, "%Y-%m-%d %H:%M:%S") - now
            left = f" ({int(d.total_seconds()//60)} min left)" if d.total_seconds() > 0 else " (EXPIRED)"
        except Exception:
            pass
    legacy = (ROOT / "runs" / "autonomy" / "AUTONOMY_ON").exists()
    print(f"ARMING (auth: autonomous_mode.json): autonomous={on}  mode={mode}  envelope_end={end}{left}")
    print(f"   legacy AUTONOMY_ON present: {legacy}  (fallback only -- arms ONLY when autonomous_mode.json is absent; "
          f"when both exist autonomous_mode.json wins, enforced in autonomy_loop.py)")

    # --- PERMISSIONS (authority: runs/autonomy/permission_policy.json gate) ---
    pol = _load("runs/autonomy/permission_policy.json")
    settings = _load(".claude/settings.json") or {}
    pre = [h.get("command") for g in settings.get("hooks", {}).get("PreToolUse", []) for h in g.get("hooks", [])]
    gate_wired = any("permission_gate" in (c or "") for c in pre)
    if pol:
        print(f"PERMISSIONS (auth: permission_policy.json gate): enabled={pol.get('enabled')}  "
              f"mode={pol.get('mode')}  file_deny={len(pol.get('file_deny_regex', []))}  "
              f"cmd_deny={len(pol.get('cmd_deny_regex', []))}  gate_wired={gate_wired}")
    else:
        print(f"PERMISSIONS: no live policy (gate falls back to normal prompts)  gate_wired={gate_wired}")

    # --- FRONTIER (the plan) ---
    fr = _load("runs/autonomy/frontier.json")
    if fr:
        nodes = fr.get("nodes", [])
        from collections import Counter
        c = Counter(n.get("status", "?") for n in nodes)
        openn = [n for n in nodes if n.get("status") == "open"]
        nxt = max(openn, key=lambda n: float(n.get("ev", 0))) if openn else None
        print(f"FRONTIER: {str(fr.get('objective',''))[:90]}...")
        print(f"   nodes: {dict(c)}  next: " + (f"{nxt.get('id')} (EV={nxt.get('ev')}) {str(nxt.get('task',''))[:60]}" if nxt else "none open"))
    else:
        print("FRONTIER: none (not armed for a frontier-driven run)")

    # --- GATES ---
    post = [h.get("command") for g in settings.get("hooks", {}).get("PostToolUse", []) for h in g.get("hooks", [])]
    meta_guard = any("meta_change_guard" in (c or "") for c in post)
    try:
        mg = subprocess.run([sys.executable, "scripts/mandatory_gate.py"], cwd=str(ROOT),
                            capture_output=True, text=True, timeout=30,
                            creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
        mg_line = (mg.stdout or "").strip().splitlines()[-1] if mg.stdout else f"exit {mg.returncode}"
    except Exception as e:
        mg_line = f"(could not run: {e})"
    print(f"GATES: mandatory_gate -> {mg_line}")
    print(f"   meta_change_guard wired (PostToolUse): {meta_guard}   (CDAP runs at commit via pre-commit hook)")

    # --- REPRODUCIBILITY + WEAK POINTS ---
    events = list(settings.get("hooks", {}).keys())
    wired_ok = {"PreToolUse", "UserPromptSubmit", "Stop"}.issubset(set(events))
    print(f"REPRODUCIBILITY: settings hook events={events}  fully_wired={wired_ok}  (bootstrap: scripts/autonomy/bootstrap.py)")
    print("WEAK POINTS (docs/SYSTEM_TOPOLOGY.md): W1 wiring=CLOSED(bootstrap)  W2 self-edit=CLOSED(meta_guard)  "
          "W3 arming/W4 perms=authority-declared  W5 prose-gap/W6 concurrency=OPEN(design)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
