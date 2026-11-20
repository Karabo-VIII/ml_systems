"""Progress monitor for in-flight pipeline refresh runs.

Until the parallel-path live-status fix lands, the parallel scheduler does
not update `data/_dag_state_live.json` between waves. This script gives a
real-time view by reading `logs/refresh/` mtimes + tailing the most-recent
log per stage. Read-only; safe to run alongside an active refresh.

Usage:
    python scripts/monitor_refresh_progress.py            # one-shot snapshot
    python scripts/monitor_refresh_progress.py --follow   # repeat every 10s

Pairs with the DAG-wave parallel scheduler shipped 2026-05-16. The serial
path already writes `_dag_state_live.json`; for serial runs prefer
`python src/pipeline/refresh.py --live`.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs" / "refresh"

TAIL_LINES = 3
RECENT_WINDOW_S = 600.0  # 10 min "actively writing" cutoff


def _tail(path: Path, n: int = TAIL_LINES) -> list[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        return [ln.rstrip() for ln in lines[-n:] if ln.strip()]
    except FileNotFoundError:
        return []


def _scan_logs() -> dict:
    if not LOG_DIR.exists():
        return {"stages": [], "error": f"{LOG_DIR} not found"}
    now = datetime.now(timezone.utc).timestamp()
    by_stage: dict[str, dict] = {}
    for p in LOG_DIR.iterdir():
        if not p.is_file() or not p.suffix == ".log" or p.name.startswith("_"):
            continue
        # Filename pattern: <stage>_<YYYYmmddTHHMMSSZ>.log
        stem = p.stem
        if "_2" not in stem:
            continue
        stage, _, ts = stem.rpartition("_")
        try:
            started = datetime.strptime(ts, "%Y%m%dT%H%M%SZ").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue
        mtime = p.stat().st_mtime
        size = p.stat().st_size
        elapsed_s = now - started.timestamp()
        age_s = now - mtime
        prev = by_stage.get(stage)
        if prev is None or started > prev["started"]:
            by_stage[stage] = {
                "stage": stage,
                "started": started,
                "log_path": p,
                "elapsed_s": elapsed_s,
                "age_s": age_s,
                "size": size,
            }
    stages = sorted(by_stage.values(), key=lambda x: x["started"])
    for s in stages:
        s["status"] = (
            "ACTIVE" if s["age_s"] < RECENT_WINDOW_S else "STALE"
        )
        if s["status"] == "ACTIVE":
            s["tail"] = _tail(s["log_path"])
    return {"stages": stages, "now": now}


def _fmt_s(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.2f}h"


def render(report: dict) -> str:
    out: list[str] = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out.append(f"=== refresh monitor @ {now} ===")
    stages = report.get("stages", [])
    if not stages:
        out.append("  no stage logs found (path: logs/refresh/)")
        return "\n".join(out)
    active = [s for s in stages if s["status"] == "ACTIVE"]
    done = [s for s in stages if s["status"] == "STALE"]
    out.append(
        f"  {len(active)} active stages | {len(done)} stale/done stages | "
        f"oldest run started {_fmt_s(stages[0]['elapsed_s'])} ago"
    )
    out.append("")
    out.append("ACTIVE (writing in last 10min):")
    if not active:
        out.append("  (none)")
    for s in active:
        out.append(
            f"  [{s['stage']:30s}] elapsed={_fmt_s(s['elapsed_s']):>6s}  "
            f"size={s['size']/1024:.0f}KB  log_age={_fmt_s(s['age_s']):>5s}"
        )
        for ln in s.get("tail", []):
            out.append(f"      | {ln[:180]}")
    out.append("")
    out.append("RECENT BUILDS (stale/done, last 12):")
    for s in done[-12:]:
        out.append(
            f"  [{s['stage']:30s}] elapsed={_fmt_s(s['elapsed_s']):>6s}  "
            f"size={s['size']/1024:.0f}KB  age={_fmt_s(s['age_s']):>5s}"
        )
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--follow", action="store_true",
                    help="Repeat every --interval seconds")
    ap.add_argument("--interval", type=int, default=10,
                    help="Refresh seconds when --follow (default 10)")
    args = ap.parse_args()
    if not args.follow:
        print(render(_scan_logs()))
        return 0
    try:
        while True:
            print("\x1b[2J\x1b[H", end="")  # clear screen
            print(render(_scan_logs()), flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
