#!/usr/bin/env python
"""watch_training.py -- a LIVE BROWSER view of the chess training (the strength curve climbing).

The trainer (az.train_robust) writes a strength curve + a log but no visual. This reads them and
renders a self-contained, auto-refreshing page (runs/train/training.html, inline SVG, no server),
opens it in your browser, and keeps re-rendering until you Ctrl-C (stopping this does NOT touch the
training). What you watch:
  * the STRENGTH CURVE -- win-rate vs random (green), vs the classical engine (orange), and the
    draw-aware CLIMB score (blue) over iterations (this is "is it learning?");
  * a header with the current iter, the champion (the net you'll PLAY), the latest gate decision,
    iters/elapsed, and the teacher depth.

  python watch_training.py            # opens the browser, refreshes ~every 4s
  python watch_training.py --once     # write the page once and exit (no loop)

No emoji (Windows cp1252). Reuses az/live_viz.py's curve renderer.
"""
from __future__ import annotations

import argparse
import html as _html
import json
import os
import re
import sys
import time
import webbrowser

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

CURVE = os.path.join(_HERE, "az", "strength_curve.json")
LOG = os.path.join(_HERE, "runs", "train", "chess_train.log")
OUT_DIR = os.path.join(_HERE, "runs", "train")
OUT = os.path.join(OUT_DIR, "training.html")


def _load_curve():
    try:
        with open(CURVE, encoding="utf-8") as f:
            rows = json.load(f)
        return rows if isinstance(rows, list) else []
    except Exception:
        return []


def _log_stats():
    """Scrape a few headline numbers from the trainer log tail. Never raises."""
    s = {"gate": "-", "champ": "-", "wall": "-", "teacher": "-", "promotes": 0, "alive_hint": ""}
    try:
        with open(LOG, encoding="utf-8", errors="replace") as f:
            raw = f.readlines()[-4000:]
    except Exception:
        return s
    # the trainer log carries huge progress-bar whitespace; normalize before matching
    tail = [re.sub(r"\s+", " ", ln).strip() for ln in raw]
    text = "\n".join(tail)
    s["promotes"] = len(re.findall(r"\bPROMOTE iter\b", text))
    for ln in reversed(tail):
        if s["gate"] == "-" and "[gate]" in ln:
            s["gate"] = re.sub(r"\s+", " ", ln.strip())[:140]
        m = re.search(r"keep champion iter (\d+)|champion iter (\d+)|PROMOTE.*iter (\d+)", ln)
        if s["champ"] == "-" and m:
            s["champ"] = next(g for g in m.groups() if g)
        m = re.search(r"wall=([0-9.]+)min", ln)
        if s["wall"] == "-" and m:
            s["wall"] = m.group(1) + " min"
        m = re.search(r"teacher depth (\d+)", ln)
        if s["teacher"] == "-" and m:
            s["teacher"] = m.group(1)
        if all(s[k] != "-" for k in ("gate", "champ", "wall", "teacher")):
            break
    return s


def _curve_html(rows):
    try:
        from az.live_viz import LiveViz
        return LiveViz._curve_svg_and_table(rows)
    except Exception as exc:  # fallback: a tiny text summary
        if not rows:
            return '<div class="muted">no strength-curve rows yet</div>'
        last = rows[-1]
        return ('<div class="muted">curve: %d rows; last iter %s, wr_random %s, climb %s '
                '(SVG renderer unavailable: %s)</div>'
                % (len(rows), last.get("iter"), last.get("winrate_vs_random"),
                   last.get("score_vs_classical_d1"), _html.escape(str(exc))))


_PAGE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">%(refresh)s
<meta name="viewport" content="width=device-width, initial-scale=1"><title>games_engine -- training</title>
<style>
 :root{--bg:#0f1117;--panel:#1a1d29;--ink:#e8eaf0;--muted:#9aa3b8;--line:#333a4d;--good:#36c08a;--accent:#f0a93b;--climb:#4d9bf0;}
 body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;line-height:1.45;}
 header{padding:16px 22px;background:var(--panel);border-bottom:1px solid var(--line);}
 header h1{margin:0 0 4px;font-size:18px;}
 .badge{font-size:10.5px;font-weight:700;padding:1px 7px;border-radius:9px;}
 .badge.live{background:#10331f;color:#36c08a;} .badge.done{background:#3a2a10;color:#f0a93b;}
 .meta{color:var(--muted);font-size:12px;margin-top:4px;}
 main{padding:22px;} .grid{display:flex;gap:34px;flex-wrap:wrap;margin-bottom:18px;}
 .stat .k{color:var(--muted);font-size:12px;} .stat .v{font-size:26px;font-weight:700;}
 .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;max-width:760px;}
 .card h3{margin:0 0 10px;font-size:14px;}
 .gate{color:var(--muted);font-size:12.5px;margin-top:14px;word-break:break-word;}
 .legend{font-size:12px;color:var(--muted);} .dot{display:inline-block;width:10px;height:10px;border-radius:50%%;vertical-align:middle;margin-right:3px;}
 .muted{color:var(--muted);font-size:13px;}
 table.curve{width:100%%;border-collapse:collapse;margin-top:10px;font-size:12.5px;}
 table.curve th,table.curve td{text-align:right;padding:3px 8px;border-bottom:1px solid var(--line);}
 table.curve th:first-child,table.curve td:first-child{text-align:left;}
 table.curve td.g,table.curve th.g{color:var(--good);} table.curve td.o,table.curve th.o{color:var(--accent);} table.curve td.b,table.curve th.b{color:var(--climb);}
 .dot.g{background:var(--good);} .dot.o{background:var(--accent);} .dot.b{background:var(--climb);}
</style></head><body>
<header><h1>games_engine -- chess training (live) %(badge)s</h1>
<div class="meta">%(ts)s &middot; champion (the net you PLAY) = iter %(champ)s &middot; promotions: %(promotes)s
 &middot; teacher depth %(teacher)s</div></header>
<main>
<div class="grid">
 <div class="stat"><div class="k">iters done</div><div class="v">%(iters)s</div></div>
 <div class="stat"><div class="k">elapsed</div><div class="v">%(wall)s</div></div>
 <div class="stat"><div class="k">wr vs random (latest)</div><div class="v" style="color:var(--good)">%(wr_rand)s</div></div>
 <div class="stat"><div class="k">climb vs classical</div><div class="v" style="color:var(--climb)">%(climb)s</div></div>
</div>
<div class="card"><h3>Strength curve (is it learning?)</h3>%(curve)s
<div class="gate"><b>latest gate:</b> %(gate)s</div></div>
<p class="muted">The champion only ever IMPROVES (gate-protected). A REJECT means the candidate did not
 beat the current champion yet -- the playable net stays as the best so far. Stopping this viewer does
 NOT stop training.</p>
</main></body></html>"""


def render(once_alive=True):
    rows = _load_curve()
    s = _log_stats()
    last = rows[-1] if rows else {}
    def fmt(x, p="{:.3f}"):
        try:
            return p.format(float(x))
        except Exception:
            return "-"
    # is training still writing? (curve mtime fresh within ~12 min)
    fresh = False
    try:
        fresh = (time.time() - os.path.getmtime(CURVE)) < 720
    except Exception:
        pass
    page = _PAGE % {
        "refresh": ("<meta http-equiv='refresh' content='4'>" if once_alive else ""),
        "badge": ("<span class='badge live'>RUNNING</span>" if fresh
                  else "<span class='badge done'>idle / stopped</span>"),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "champ": _html.escape(str(s["champ"])), "promotes": s["promotes"],
        "teacher": _html.escape(str(s["teacher"])),
        "iters": (str(last.get("iter", len(rows))) if rows else "0"),
        "wall": _html.escape(str(s["wall"])),
        "wr_rand": fmt(last.get("winrate_vs_random")),
        "climb": fmt(last.get("score_vs_classical_d1")),
        "curve": _curve_html(rows),
        "gate": _html.escape(str(s["gate"])),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(page)
    os.replace(tmp, OUT)
    return OUT


def main():
    ap = argparse.ArgumentParser(description="Live browser view of the chess training strength curve.")
    ap.add_argument("--once", action="store_true", help="write the page once and exit (no loop/refresh)")
    ap.add_argument("--interval", type=float, default=4.0, help="seconds between refreshes (default 4)")
    args = ap.parse_args()

    path = render(once_alive=not args.once)
    print(f"[train-viz] live view: file://{path.replace(os.sep, '/')}")
    if args.once:
        return 0
    if os.environ.get("GAMESENGINE_NO_BROWSER", "") not in ("1", "true", "TRUE"):
        try:
            webbrowser.open("file://" + path.replace(os.sep, "/"))
        except Exception:
            pass
    print("[train-viz] refreshing; Ctrl-C to stop (training keeps running).")
    try:
        while True:
            time.sleep(max(1.0, args.interval))
            render(once_alive=True)
    except KeyboardInterrupt:
        print("\n[train-viz] stopped (training unaffected).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
