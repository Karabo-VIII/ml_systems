#!/usr/bin/env python3
"""dashboard.py -- a NEAR-REAL-TIME visual window into the harness LEARNING (the user's ask 2026-06-06: "an NRT visual
tool to show progress of learning, for visual validation, or make it replayable").

The image the user shared says it all: the model is fixed; we make the HARNESS tighter. This renders what that harness
is DOING and LEARNING, from the real state files -- no model in the loop, pure observability:
  - FRONTIER (frontier.json): the EV-ranked nodes + their status (open/in_progress/done/blocked) = what is being worked
  - LEARNING LANES (runs/autonomy/learnings/*.jsonl): the lessons the loops wrote = what was learned
  - FULFILLMENT LEDGER (frontier.overseer.fulfillment_ledger): the verdict timeline = learning PROGRESS over time (the
    REPLAYABLE trace -- each row is a node judged, in order, with its evidence + SHA + date)
  - WATCHER (watcher.log): liveness heartbeat
  - SKILL LIBRARY (skill_library/INDEX.json): capabilities harvested (the monotonic growth)

Writes a self-contained dark-theme HTML (no external deps) with a meta-refresh for NRT. `--watch N` regenerates every N
seconds (open the HTML in a browser and watch it move). The fulfillment ledger IS the replay -- it is an ordered,
timestamped trace of every learning event. No emoji in code (cp1252).

Usage:
  python scripts/autonomy/dashboard.py                 # generate runs/autonomy/dashboard.html once
  python scripts/autonomy/dashboard.py --watch 5       # regenerate every 5s (NRT); HTML auto-refreshes too
"""
from __future__ import annotations

import glob
import html
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AUT = os.path.join(ROOT, "runs", "autonomy")
OUT = os.path.join(AUT, "dashboard.html")
STATUS_COLOR = {"open": "#5a9bd4", "in_progress": "#e0a458", "done": "#5cb85c", "blocked": "#d9534f",
                "refuted": "#d9534f", "pass": "#5cb85c"}


def _load_json(p, default):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return default


def _lane_recent(lane, n=4):
    p = os.path.join(AUT, "learnings", f"{lane}.jsonl")
    if not os.path.exists(p):
        return 0, []
    rows = [l for l in open(p, encoding="utf-8") if l.strip()]
    recent = []
    for l in rows[-n:]:
        try:
            r = json.loads(l)
            recent.append((r.get("lesson") or r.get("text") or "")[:160])
        except Exception:
            continue
    return len(rows), recent


def _watcher_state():
    p = os.path.join(AUT, "watcher.log")
    if not os.path.exists(p):
        return "no watcher", 999
    age = (time.time() - os.path.getmtime(p)) / 60.0
    last = ""
    try:
        last = open(p, encoding="utf-8").read().strip().splitlines()[-1][:80]
    except Exception:
        pass
    return last, age


def render(now_str: str) -> str:
    fr = _load_json(os.path.join(AUT, "frontier.json"), {})
    nodes = fr.get("nodes", [])
    obj = fr.get("objective", "(no objective)")
    ledger = fr.get("overseer", {}).get("fulfillment_ledger", [])
    n_open = sum(1 for n in nodes if n.get("status") == "open")
    n_done = sum(1 for n in nodes if n.get("status") == "done")
    wlast, wage = _watcher_state()
    wcolor = "#5cb85c" if wage < 3 else "#d9534f"
    skills = _load_json(os.path.join(AUT, "skill_library", "INDEX.json"), {})
    n_skills = len(skills.get("skills", skills)) if isinstance(skills, dict) else 0

    def esc(s):
        return html.escape(str(s))

    # frontier rows (EV-ranked)
    fr_rows = ""
    for nd in sorted(nodes, key=lambda x: (x.get("status") != "open", -x.get("ev", 0))):
        c = STATUS_COLOR.get(nd.get("status", "open"), "#888")
        fr_rows += (f"<tr><td style='color:{c};font-weight:bold'>{esc(nd.get('status','')).upper()}</td>"
                    f"<td>{nd.get('ev','')}</td><td><b>{esc(nd.get('id',''))}</b></td>"
                    f"<td>{esc(nd.get('task',''))[:120]}</td></tr>")

    # learning lanes
    lane_html = ""
    for lane in ["expert", "plain", "meta", "sol"]:
        cnt, recent = _lane_recent(lane)
        items = "".join(f"<li>{esc(x)}</li>" for x in recent)
        lane_html += f"<div class=lane><h4>{lane} <span class=cnt>{cnt}</span></h4><ul>{items}</ul></div>"

    # fulfillment ledger = the replayable learning trace (newest first)
    led_rows = ""
    for e in reversed(ledger[-25:]):
        v = e.get("verdict", "")
        c = "#5cb85c" if "PASS" in v.upper() else ("#d9534f" if "REFUT" in v.upper() else "#e0a458")
        led_rows += (f"<tr><td>{esc(e.get('date',''))[:19]}</td><td style='color:{c};font-weight:bold'>{esc(v)}</td>"
                     f"<td><b>{esc(e.get('node',''))}</b></td><td>{esc(e.get('evidence',''))[:140]}</td>"
                     f"<td class=sha>{esc(e.get('sha',''))}</td></tr>")

    return f"""<!doctype html><html><head><meta charset=utf-8>
<meta http-equiv=refresh content=8>
<title>Harness Learning -- NRT</title>
<style>
body{{background:#0d1117;color:#c9d1d9;font:13px/1.5 -apple-system,Segoe UI,sans-serif;margin:0;padding:18px}}
h1{{font-size:18px;margin:0 0 4px;color:#e6edf3}} h3{{color:#8b949e;margin:18px 0 6px;font-size:13px;text-transform:uppercase;letter-spacing:.5px}}
.sub{{color:#8b949e;font-size:12px;margin-bottom:10px}}
.cards{{display:flex;gap:12px;margin:8px 0}} .card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px 14px;min-width:90px}}
.card .v{{font-size:22px;font-weight:bold}} .card .l{{color:#8b949e;font-size:11px}}
table{{width:100%;border-collapse:collapse;font-size:12px}} td{{padding:4px 8px;border-bottom:1px solid #21262d;vertical-align:top}}
th{{text-align:left;color:#8b949e;padding:4px 8px;font-weight:normal}}
.lanes{{display:flex;gap:10px;flex-wrap:wrap}} .lane{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:8px 12px;flex:1;min-width:220px}}
.lane h4{{margin:0 0 4px;color:#e6edf3}} .lane .cnt{{background:#30363d;border-radius:10px;padding:0 7px;font-size:11px;color:#8b949e}}
.lane ul{{margin:4px 0 0;padding-left:16px}} .lane li{{color:#8b949e;margin-bottom:3px}}
.sha{{font-family:monospace;color:#6e7681}} .dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}}
</style></head><body>
<h1>Harness Learning &mdash; near-real-time</h1>
<div class=sub><span class=dot style='background:{wcolor}'></span>watcher {wage:.1f}m ago &middot; {esc(wlast)} &middot; rendered {now_str} &middot; auto-refresh 8s</div>
<div class=sub><b>Objective:</b> {esc(obj)[:200]}</div>
<div class=cards>
 <div class=card><div class=v>{len(nodes)}</div><div class=l>frontier nodes</div></div>
 <div class=card><div class=v style='color:#5a9bd4'>{n_open}</div><div class=l>open</div></div>
 <div class=card><div class=v style='color:#5cb85c'>{n_done}</div><div class=l>done</div></div>
 <div class=card><div class=v>{len(ledger)}</div><div class=l>verdicts</div></div>
 <div class=card><div class=v>{n_skills}</div><div class=l>skills harvested</div></div>
</div>
<h3>Frontier &mdash; what is being worked (EV-ranked)</h3>
<table><tr><th>status</th><th>ev</th><th>id</th><th>task</th></tr>{fr_rows}</table>
<h3>Learning lanes &mdash; what was learned</h3>
<div class=lanes>{lane_html}</div>
<h3>Fulfillment ledger &mdash; the replayable learning trace (newest first)</h3>
<table><tr><th>when</th><th>verdict</th><th>node</th><th>evidence</th><th>sha</th></tr>{led_rows}</table>
</body></html>"""


def generate():
    os.makedirs(AUT, exist_ok=True)
    now_str = time.strftime("%H:%M:%S")
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(render(now_str))
    os.replace(tmp, OUT)
    return OUT


def main():
    if "--watch" in sys.argv:
        i = sys.argv.index("--watch")
        interval = int(sys.argv[i + 1]) if i + 1 < len(sys.argv) else 5
        print(f"[dashboard] NRT mode: regenerating {OUT} every {interval}s (Ctrl-C to stop). Open it in a browser.")
        try:
            while True:
                generate()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("[dashboard] stopped")
        return 0
    p = generate()
    print(f"[dashboard] wrote {p} -- open it in a browser (auto-refreshes every 8s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
