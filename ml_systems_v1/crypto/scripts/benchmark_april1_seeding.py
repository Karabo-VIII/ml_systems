"""Benchmark seeding from 2026-04-01 (user-directed, 2026-04-25 session).

Cold-boots paper_trader_v2 from 2026-04-01 for every PROFILE in the registry
that does not already have an April-1 benchmark seed. Special focus on sub-day
profiles per user request.

Why:
  Project has many seeds initialized from 2025-01-01 (long replay) and
  staggered start dates (Mar 17, Apr 2, etc.). For honest comparison of
  recent-window performance, all candidates need a seed initialized from
  exactly 2026-04-01 = "April benchmark" reference window.

Usage:
  python scripts/benchmark_april1_seeding.py                  # all profiles
  python scripts/benchmark_april1_seeding.py --subday-only    # subday only
  python scripts/benchmark_april1_seeding.py --core-only      # top-12 deployables only
  python scripts/benchmark_april1_seeding.py --dry-run        # plan only
  python scripts/benchmark_april1_seeding.py --parallel 4     # higher concurrency

Output:
  logs/paper_trader_v2/seeds/<profile>_apr1/daily_snapshot.csv
  logs/april1_benchmark_2026_04_25/run_results.json
  logs/april1_benchmark_2026_04_25/SUMMARY.md
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
PAPER_TRADER = ROOT / "src" / "analysis" / "paper_trader_v2.py"
SEEDS_DIR = ROOT / "logs" / "paper_trader_v2" / "seeds"
LOG_DIR = ROOT / "logs" / "april1_benchmark_2026_04_25"
LOG_DIR.mkdir(parents=True, exist_ok=True)

FROM_DATE = "2026-04-01"
SEED_SUFFIX = "_apr1"
CAPITAL = 10000.0


def get_all_profiles() -> list[str]:
    """Load all profile names from src/strategy/strat_profiles.PROFILES."""
    sys.path.insert(0, str(ROOT))
    from src.strategy import strat_profiles as sp  # type: ignore
    return list(sp.PROFILES.keys())


# Top-12 deployable shortlist (from memory/alignment_2026_04_23.md + 2026-04-25 DSR audit)
CORE_DEPLOYABLES = [
    "prod_meta_combined", "prod_meta_full", "prod_meta_combined_k15",
    "prod_meta_full_k15", "prod_meta_medium", "prod_meta_short",
    # Below are aliases / DEAD profiles preserved for completeness but tagged
    "prod_combined", "prod_swing", "prod_short", "prod_medium", "prod_trend",
    "prod_meta_plus_subday",
]

# Sub-day profiles (user's special focus)
SUBDAY_PROFILES = [
    "subday_oversold", "subday_vpin_dip", "subday_shallow", "subday_funding",
    "subday_combined", "subday_vpin_flow", "subday_vol_break",
    "prod_subday_s1", "prod_subday_s2", "prod_subday_s3", "prod_subday_s4",
    "prod_subday_s5", "prod_subday_all", "prod_meta_plus_subday",
]


def run_one(profile: str) -> dict:
    """Reset + cold-boot one profile from 2026-04-01."""
    seed_name = f"{profile}{SEED_SUFFIX}"
    t0 = time.time()
    result = {"profile": profile, "seed": seed_name, "status": "pending"}

    # Reset prior state
    try:
        subprocess.call(
            [PY, str(PAPER_TRADER), "--reset", "--seed", seed_name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        result["status"] = "reset_error"
        result["err"] = str(e)
        return result

    # Init from 2026-04-01
    try:
        cmd = [
            PY, str(PAPER_TRADER), "--init",
            "--seed", seed_name,
            "--profile", profile,
            "--capital", str(CAPITAL),
            "--from-date", FROM_DATE,
            "--skip-refresh",
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        result["rc"] = out.returncode
        result["dt"] = round(time.time() - t0, 1)
        for line in out.stdout.splitlines():
            if "Equity=$" in line:
                result["final_equity_raw"] = line.strip()
            if "TOTAL :" in line:
                result["total_line"] = line.strip()
        result["status"] = "ok" if out.returncode == 0 else "init_failed"
        if out.returncode != 0:
            tail = (out.stdout + "\n" + out.stderr).splitlines()[-30:]
            result["tail"] = "\n".join(tail)
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["dt"] = round(time.time() - t0, 1)
    except Exception as e:
        result["status"] = "exception"
        result["err"] = str(e)
    return result


def already_has_april_seed(profile: str) -> bool:
    """Check if <profile>_apr1 already has a daily_snapshot starting 2026-04-01."""
    snap = SEEDS_DIR / f"{profile}{SEED_SUFFIX}" / "daily_snapshot.csv"
    if not snap.exists():
        return False
    try:
        with open(snap, "r") as f:
            header = f.readline()
            first_row = f.readline()
            if not first_row:
                return False
            first_date = first_row.split(",")[0].strip()
            return first_date >= "2026-03-31"  # accept Mar 31 or Apr 1
    except Exception:
        return False


def parse_final_equity(seed_name: str) -> tuple[float | None, int]:
    """Read daily_snapshot.csv, return (final_equity, n_days)."""
    snap = SEEDS_DIR / seed_name / "daily_snapshot.csv"
    if not snap.exists():
        return None, 0
    try:
        import polars as pl
        df = pl.read_csv(snap, infer_schema_length=1000)
        if "total_equity" not in df.columns:
            return None, 0
        return float(df["total_equity"][-1]), len(df)
    except Exception:
        return None, 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--subday-only", action="store_true")
    ap.add_argument("--core-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip profiles that already have <profile>_apr1 starting 2026-04-01")
    ap.add_argument("--only", nargs="+", help="Limit to these profiles")
    args = ap.parse_args()

    if args.subday_only:
        candidates = SUBDAY_PROFILES
    elif args.core_only:
        candidates = CORE_DEPLOYABLES + SUBDAY_PROFILES
    else:
        candidates = get_all_profiles()

    if args.only:
        candidates = [p for p in candidates if p in args.only]

    if args.skip_existing:
        before = len(candidates)
        candidates = [p for p in candidates if not already_has_april_seed(p)]
        print(f"[skip-existing] {before - len(candidates)} already have April seeds")

    print("=" * 80)
    print(f"APRIL-1 BENCHMARK SEEDING -- 2026-04-25")
    print(f"From: {FROM_DATE}    Capital: ${CAPITAL:,.0f}    Suffix: {SEED_SUFFIX}")
    print(f"Profiles: {len(candidates)}    Parallelism: {args.parallel}")
    print(f"UTC: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 80)
    for p in candidates:
        marker = "(subday)" if p in SUBDAY_PROFILES else ""
        print(f"  {p:<40} -> {p}{SEED_SUFFIX}  {marker}")

    if args.dry_run:
        print("\n(dry-run; nothing executed)")
        return

    if not candidates:
        print("\nNothing to do.")
        return

    results = []
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.parallel) as ex:
        futures = {ex.submit(run_one, p): p for p in candidates}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"profile": p, "seed": f"{p}{SEED_SUFFIX}", "status": "exception_outer", "err": str(e)}
            # Summary fields
            fe, n_days = parse_final_equity(r["seed"])
            r["final_equity"] = fe
            r["n_days_recorded"] = n_days
            r["benchmark_pct"] = (fe / CAPITAL - 1) * 100 if fe else None
            results.append(r)
            t_elapsed = time.time() - t_start
            elap_min = t_elapsed / 60
            ok = sum(1 for x in results if x["status"] == "ok")
            fail = len(results) - ok
            tag = ("OK " + (f"+{r['benchmark_pct']:.2f}%" if r.get("benchmark_pct") is not None else "")
                   if r["status"] == "ok"
                   else r["status"])
            print(f"[{len(results):>3}/{len(candidates)}] {p:<35} {tag:<25} ({r.get('dt', '?')}s) "
                  f"| {ok} ok, {fail} fail | elapsed {elap_min:.1f}m")

    # Persist results
    out_json = LOG_DIR / "run_results.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))

    # SUMMARY.md
    ok_runs = sorted([r for r in results if r["status"] == "ok"],
                     key=lambda r: r.get("benchmark_pct") or -1e9, reverse=True)
    fail_runs = [r for r in results if r["status"] != "ok"]

    md = []
    md.append(f"# April 1 Benchmark Seeding -- 2026-04-25")
    md.append("")
    md.append(f"- Window: 2026-04-01 -> latest bar")
    md.append(f"- Capital: ${CAPITAL:,.0f} per profile")
    md.append(f"- Profiles attempted: {len(candidates)}")
    md.append(f"- Successful: {len(ok_runs)}")
    md.append(f"- Failed: {len(fail_runs)}")
    md.append("")
    md.append(f"## Top 25 by April-1 benchmark return")
    md.append("")
    md.append("| Profile | Equity | Return | Days | Time |")
    md.append("|---------|-------|--------|------|------|")
    for r in ok_runs[:25]:
        eq = r.get("final_equity")
        bp = r.get("benchmark_pct")
        eq_s = f"${eq:,.2f}" if eq else "--"
        bp_s = f"{bp:+.2f}%" if bp is not None else "--"
        md.append(f"| {r['profile']} | {eq_s} | {bp_s} | {r.get('n_days_recorded', 0)} | {r.get('dt', '?')}s |")

    md.append("")
    md.append(f"## Sub-day profiles (user-emphasized)")
    md.append("")
    md.append("| Profile | Equity | Return | Days |")
    md.append("|---------|-------|--------|------|")
    for r in [x for x in results if x["profile"] in SUBDAY_PROFILES]:
        eq = r.get("final_equity")
        bp = r.get("benchmark_pct")
        eq_s = f"${eq:,.2f}" if eq else "--"
        bp_s = f"{bp:+.2f}%" if bp is not None else r["status"]
        md.append(f"| {r['profile']} | {eq_s} | {bp_s} | {r.get('n_days_recorded', 0)} |")

    if fail_runs:
        md.append("")
        md.append("## Failures")
        md.append("")
        md.append("| Profile | Status | Note |")
        md.append("|---------|--------|------|")
        for r in fail_runs:
            note = r.get("err", r.get("tail", ""))[:200]
            md.append(f"| {r['profile']} | {r['status']} | {note} |")

    (LOG_DIR / "SUMMARY.md").write_text("\n".join(md))

    print()
    print(f"[apr1] saved: {out_json}")
    print(f"[apr1] summary: {LOG_DIR/'SUMMARY.md'}")
    print(f"[apr1] DONE: {len(ok_runs)}/{len(candidates)} ok in {(time.time()-t_start)/60:.1f}m")


if __name__ == "__main__":
    main()
