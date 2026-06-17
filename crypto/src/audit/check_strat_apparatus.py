"""CDAP Layer 9 -- STRAT APPARATUS regression gate (closes the 2026-06-05 preflight gap).

The measurement apparatus (src/strat: battery / firewall / positive_control / dsr) is sound NOW (its selftest
proves two-sided power: accepts a known edge, rejects ghosts/beta), but nothing GUARDS it against a future
regression -- a broken apparatus would make every strategy verdict untrustworthy (the exact failure that caused
the 2026-06-04 reset). This gate runs `src/strat/selftest_all.py` on every commit; if the apparatus loses power
or a selftest breaks, it returns exit 2 (HALT). Interface mirrors check_chimera_liveness / check_dsr_holm:
run_audit() -> (list[dict(severity,name,file,detail)], exit_code). No emoji (Windows cp1252).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SELFTEST = PROJECT_ROOT / "src" / "strat" / "selftest_all.py"


def run_audit() -> tuple[list[dict], int]:
    if not SELFTEST.exists():
        return ([{
            "severity": "warn",
            "name": "strat_apparatus_selftest_missing",
            "file": "src/strat/selftest_all.py",
            "detail": "src/strat/selftest_all.py not found -- the measurement apparatus is NOT regression-guarded.",
        }], 1)
    try:
        r = subprocess.run([sys.executable, str(SELFTEST)], cwd=str(PROJECT_ROOT),
                           capture_output=True, text=True, timeout=180)
    except Exception as e:
        return ([{
            "severity": "warn",
            "name": "strat_apparatus_selftest_error",
            "file": "src/strat/selftest_all.py",
            "detail": f"could not run the apparatus selftest: {e}",
        }], 1)
    if r.returncode != 0:
        tail = "\n".join(((r.stdout or "") + (r.stderr or "")).strip().splitlines()[-6:])
        return ([{
            "severity": "critical",
            "name": "strat_apparatus_regression",
            "file": "src/strat/",
            "detail": (
                f"src/strat/selftest_all.py FAILED (exit {r.returncode}) -- the measurement apparatus "
                f"(battery/firewall/positive_control/dsr) lost power or broke. A regression here makes EVERY "
                f"strategy verdict untrustworthy (the 2026-06-04-reset failure mode). HALT; fix the apparatus.\n{tail}"
            ),
        }], 2)
    return ([{
        "severity": "info",
        "name": "strat_apparatus_ok",
        "file": "src/strat/",
        "detail": "src/strat apparatus selftest passed -- two-sided power confirmed (accepts edge, rejects ghost/beta), leak-clean.",
    }], 0)


def main() -> int:
    findings, exit_code = run_audit()
    for f in findings:
        print(f"[check_strat_apparatus] {f['severity'].upper()} {f['name']}: {f['detail'][:200]}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
