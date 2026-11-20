#!/usr/bin/env python
"""soak_run.py -- run ALL engines flat out (max speed: no delay, no render) for N hours,
NEVER stopping on a single-game error. This is the unattended robustness/soak harness.

It plays "the 4 in a row" -- chess (champion self-play), connect-4 (net vs heuristic),
atari (MinAtar, rotating breakout/space_invaders/asterix), catch (MuZero) -- back to back,
loops until the wall-clock window closes, wraps every game in try/except so a single failure
logs and the loop CONTINUES, and records cumulative throughput + per-engine W/L + RSS memory
(leak watch) to runs/soak/soak_log.txt and a glanceable runs/soak/status.html.

  python soak_run.py                 # 2 hours, max speed (the default)
  python soak_run.py --hours 1.5
  python soak_run.py --max-iters 1   # one cycle of all 4 (smoke test), ignores the clock

No emoji (Windows cp1252). Bounded + resumable-by-relaunch; the status HTML auto-refreshes.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import platform
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import run_engines as RE
from run_engines import _resolve_device


def _fmt_hms(secs: float) -> str:
    secs = int(max(0, secs))
    return f"{secs // 3600:d}h{(secs % 3600) // 60:02d}m{secs % 60:02d}s"


def write_status_html(path, stats, totals, started_str, elapsed, hours, mem_mb, done=False):
    rows = ""
    for name in ("chess", "connect4", "minatar", "catch"):
        s = stats.get(name)
        if not s:
            continue
        wl = (f"W{s['wins']} D{s['draws']} L{s['losses']}" if (s['wins'] or s['draws'] or s['losses'])
              else (f"mean score {s['score'] / s['runs']:.1f}" if s['runs'] else "-"))
        rows += (f"<tr><td>{name}</td><td>{s['runs']}</td><td>{wl}</td>"
                 f"<td class='{'err' if s['errors'] else 'ok'}'>{s['errors']}</td></tr>")
    gps = totals['games'] / elapsed if elapsed > 0 else 0.0
    remaining = max(0.0, hours * 3600 - elapsed)
    badge = ("<span class='b done'>SOAK COMPLETE</span>" if done
             else "<span class='b live'>RUNNING</span>")
    refresh = "" if done else "<meta http-equiv='refresh' content='5'>"
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">{refresh}
<title>games_engine soak</title><style>
 body{{margin:0;background:#0f1117;color:#e8eaf0;font-family:Segoe UI,Arial,sans-serif}}
 header{{padding:16px 22px;background:#1a1d29;border-bottom:1px solid #333a4d}}
 h1{{margin:0 0 4px;font-size:18px}} .meta{{color:#9aa3b8;font-size:12px}}
 .b{{font-size:11px;font-weight:700;padding:1px 7px;border-radius:9px}}
 .b.live{{background:#10331f;color:#36c08a}} .b.done{{background:#3a2a10;color:#f0a93b}}
 main{{padding:22px}} table{{border-collapse:collapse;font-size:14px;min-width:380px}}
 th,td{{text-align:left;padding:6px 14px;border-bottom:1px solid #333a4d}}
 td.err{{color:#e2463f;font-weight:700}} td.ok{{color:#36c08a}}
 .big{{font-size:30px;font-weight:700;margin:6px 0}} .grid{{display:flex;gap:40px;flex-wrap:wrap}}
</style></head><body>
<header><h1>games_engine -- 2h max-speed soak {badge}</h1>
<div class="meta">started {started_str} &middot; elapsed {_fmt_hms(elapsed)} / {hours:.1f}h &middot;
 ~{_fmt_hms(remaining)} left &middot; RSS {mem_mb}</div></header>
<main><div class="grid">
 <div><div class="meta">total games</div><div class="big">{totals['games']}</div></div>
 <div><div class="meta">errors (loop never stops)</div><div class="big" style="color:{'#e2463f' if totals['errors'] else '#36c08a'}">{totals['errors']}</div></div>
 <div><div class="meta">throughput</div><div class="big">{gps:.2f}<span style="font-size:14px"> games/s</span></div></div>
</div>
<table><thead><tr><th>engine</th><th>games</th><th>record</th><th>errors</th></tr></thead>
<tbody>{rows}</tbody></table>
<p class="meta">Max speed (no delay, no render). A per-game crash is logged and the loop continues.
 Full log: runs/soak/soak_log.txt</p></main></body></html>"""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(html)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run all engines flat out for N hours (soak/robustness).")
    ap.add_argument("--hours", type=float, default=2.0, help="wall-clock window (default 2.0)")
    ap.add_argument("--max-iters", type=int, default=0, help="stop after N cycles instead of the clock (smoke test)")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    dev = _resolve_device(args.device)
    out_dir = os.path.join(_HERE, "runs", "soak")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "soak_log.txt")
    status_path = os.path.join(out_dir, "status.html")

    try:
        import psutil
        proc = psutil.Process()
    except Exception:
        proc = None

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    minatar_games = RE._available_minatar_games() or ["breakout"]
    started_str = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.time()
    end = t0 + args.hours * 3600
    log(f"SOAK START -- hours={args.hours} device={dev} host={platform.node()} "
        f"minatar={minatar_games} max_iters={args.max_iters or 'off'}")

    stats = {}
    totals = {"games": 0, "errors": 0}
    mem_mb = "n/a"
    it = 0
    mi = 0

    def run_one(name, fn):
        s = stats.setdefault(name, dict(runs=0, errors=0, wins=0, draws=0, losses=0, score=0.0))
        try:
            with contextlib.redirect_stdout(io.StringIO()):  # silence the verbose per-game prints
                out = fn()
            s["runs"] += 1
            totals["games"] += 1
            if isinstance(out, dict):
                s["wins"] += out.get("w", 0); s["draws"] += out.get("d", 0); s["losses"] += out.get("l", 0)
                if "mean_return" in out:
                    s["score"] += float(out.get("mean_return", 0.0))
        except Exception as e:
            s["errors"] += 1
            totals["errors"] += 1
            log(f"ERROR {name} (iter {it}): {type(e).__name__}: {e}")
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(traceback.format_exc() + "\n")
            except Exception:
                pass

    while time.time() < end:
        it += 1
        g = minatar_games[mi % len(minatar_games)]
        mi += 1
        # "the 4 in a row" at max speed (no delay, no render, no web)
        run_one("chess", lambda: RE.play_chess(games=1, delay=0, render=False, device=dev, mcts_sims=16))
        if time.time() >= end: break
        run_one("connect4", lambda: RE.play_connect4(games=1, delay=0, render=False, device=dev, mcts_sims=32))
        if time.time() >= end: break
        run_one("minatar", lambda: RE.play_atari_minatar(games=1, delay=0, render=False, device=dev,
                                                          max_steps=1000, max_render_steps=0, game=g))
        if time.time() >= end: break
        run_one("catch", lambda: RE.play_atari(games=1, delay=0, render=False, device=dev, mcts_sims=12))

        elapsed = time.time() - t0
        if proc is not None:
            try:
                mem_mb = f"{proc.memory_info().rss / 1e6:.0f}MB"
            except Exception:
                pass
        gps = totals["games"] / elapsed if elapsed > 0 else 0.0
        log(f"iter {it}: {totals['games']} games, {totals['errors']} errors, {gps:.2f} games/s, "
            f"rss={mem_mb}, elapsed={_fmt_hms(elapsed)}/{args.hours:.1f}h")
        write_status_html(status_path, stats, totals, started_str, elapsed, args.hours, mem_mb)

        if args.max_iters and it >= args.max_iters:
            log(f"max-iters={args.max_iters} reached -- stopping (smoke mode)")
            break

    elapsed = time.time() - t0
    write_status_html(status_path, stats, totals, started_str, elapsed, args.hours, mem_mb, done=True)
    log(f"SOAK DONE -- {totals['games']} games, {totals['errors']} errors over {_fmt_hms(elapsed)}. "
        f"per-engine={ {k: dict(runs=v['runs'], errors=v['errors']) for k, v in stats.items()} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
