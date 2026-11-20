"""Align every PROD-candidate strategy snapshot to fresh chimera data.

For each profile in the curated list, either:
    - Reset + cold-boot from 2025-01-01 via paper_trader_v2 --init
      (produces pt_<seed>/daily_snapshot.csv through latest bar)
    - If script-driven sleeve, run the regen script.

Runs profiles in parallel with a concurrency cap to keep RAM in check.

Usage:
    python scripts/align_all_prod_candidates.py              # run all, concurrency 3
    python scripts/align_all_prod_candidates.py --parallel 1  # sequential
    python scripts/align_all_prod_candidates.py --dry-run    # plan only
    python scripts/align_all_prod_candidates.py --skip-existing  # don't redo fresh snapshots
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

# Profiles to align. Each tuple is (profile_name, seed_dir_name, notes).
# Seed_dir matches what portfolio_aggregator.py expects when the profile is listed
# in a blend; otherwise it's just a storage location.
TARGET_PROFILES = [
    # Tier S/A ranked entries not yet aligned
    ("prod_meta_medium",         "pt_meta_medium",          "Tier-A Rank 5 singleton (never seeded post-fix)"),
    ("prod_meta_combined_k15",   "pt_meta_combined_k15",    "Alt sizing (Kelly k=1.5) of Rank 1 champion"),
    ("prod_meta_full_k15",       "pt_meta_full_k15",        "Alt sizing of Rank 3 prod_meta_full"),
    ("prod_meta_short",          "pt_meta_short",           "Short-only meta-labeler variant"),
    # Ranked tier base sleeves
    ("prod_combined",            "pt_prod_combined",        "Base combined book"),
    ("prod_swing",               "pt_prod_swing",           "Base swing book"),
    ("prod_short",               "pt_prod_short",           "Base short book"),
    ("prod_medium",              "pt_prod_medium",          "Base medium (trend) book"),
    ("prod_trend",               "pt_prod_trend",           "Base trend book"),
    # Floor sizing variants (pt proxy for yaml-ranked size13_prod_floor_*)
    ("prod_floor_combined",      "pt_prod_floor_combined",  "Floor sizing combined"),
    ("prod_floor_medium",        "pt_prod_floor_medium",    "Floor sizing medium"),
    # DEAD entries (align for completeness)
    ("perp_dna_long_short",      "pt_perp_dna",             "DEAD: bear regime, dropped from deploy"),
    ("regime_routed_full",       "pt_regime_routed",        "DEAD: bull-killed regime overlay"),
    # Missing / ranked tier
    ("multi_prod_combined",      "pt_multi_prod_combined",  "Rank 8 multi-stream combined"),
    ("kelly_vol_break",          "pt_kelly_vol_break",      "Rank 14, previously missing"),
    # Size13 / v4 variants
    ("prod_size_xsec",           "pt_prod_size_xsec",       "Size13 xsec variant"),
    # Stream / brain / capture
    ("multi_stream",             "pt_multi_stream",         "Multi-stream variant"),
    ("capture_v1_conservative",  "pt_capture_v1_cons",      "Capture v1 conservative"),
    # Tactical overlays
    ("subday_combined",          "pt_subday_combined",      "Subday bundled (expected degraded)"),
    # Research-track
    ("prod_v7sd_combined",       "pt_v7sd_combined",        "v7 signal-direction combined"),
    ("prod_te_leadlag",          "pt_te_leadlag",           "Time-shuffled lead-lag"),
]


def run_one(profile: str, seed: str) -> dict:
    """Reset + cold-boot one profile. Returns result dict."""
    t0 = time.time()
    result = {"profile": profile, "seed": seed, "status": "pending"}

    # Reset first (clear prior state)
    try:
        subprocess.call([PY, str(PAPER_TRADER), "--reset", "--seed", seed],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        result["status"] = "reset_error"
        result["err"] = str(e)
        return result

    # Init from 2025-01-01
    try:
        cmd = [PY, str(PAPER_TRADER), "--init",
               "--seed", seed,
               "--profile", profile,
               "--capital", "10000",
               "--from-date", "2025-01-01",
               "--skip-refresh"]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        result["rc"] = out.returncode
        result["dt"] = round(time.time() - t0, 1)
        # Scrape equity + trade count
        stdout = out.stdout
        for line in stdout.splitlines():
            if "Equity=$" in line:
                result["final_equity_raw"] = line.strip()
            if "TOTAL :" in line:
                result["total_line"] = line.strip()
        result["status"] = "ok" if out.returncode == 0 else "init_failed"
        if out.returncode != 0:
            # Capture last 30 lines of stderr/stdout for debug
            tail = (stdout + "\n" + out.stderr).splitlines()[-30:]
            result["tail"] = "\n".join(tail)
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["dt"] = round(time.time() - t0, 1)
    except Exception as e:
        result["status"] = "exception"
        result["err"] = str(e)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parallel", type=int, default=3, help="Concurrent profiles (default 3)")
    ap.add_argument("--dry-run", action="store_true", help="Print plan, do not run")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip profiles whose seed has a snapshot dated >= 2026-04-20")
    ap.add_argument("--only", nargs="+", help="Run only these profiles")
    args = ap.parse_args()

    if args.only:
        targets = [(p, s, n) for (p, s, n) in TARGET_PROFILES if p in args.only]
    else:
        targets = list(TARGET_PROFILES)

    if args.skip_existing:
        import pandas as pd
        filtered = []
        for (p, s, n) in targets:
            snap = SEEDS_DIR / s / "daily_snapshot.csv"
            if snap.exists():
                try:
                    df = pd.read_csv(snap, usecols=["date"])
                    last = pd.to_datetime(df["date"]).max()
                    if last.date().isoformat() >= "2026-04-20":
                        print(f"  [skip-existing] {p:<30} last={last.date()}")
                        continue
                except Exception:
                    pass
            filtered.append((p, s, n))
        targets = filtered

    print("=" * 90)
    print(f"ALIGN PROD CANDIDATES -- {len(targets)} profiles, parallelism={args.parallel}")
    print(f"UTC: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 90)
    for (p, s, n) in targets:
        print(f"  {p:<30} -> seeds/{s:<30} {n}")

    if args.dry_run:
        print("\n(dry-run; nothing executed)")
        return

    results = []
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.parallel) as ex:
        futures = {ex.submit(run_one, p, s): (p, s) for (p, s, _) in targets}
        for fut in as_completed(futures):
            p, s = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"profile": p, "seed": s, "status": "exception_outer", "err": str(e)}
            results.append(r)
            t_elapsed = time.time() - t_start
            status = r.get("status", "?")
            dt = r.get("dt", "?")
            total = r.get("total_line", r.get("final_equity_raw", ""))[:70]
            print(f"  [{len(results):>2}/{len(targets)}]  {p:<30} {status:<12} dt={dt}s  elapsed={int(t_elapsed)}s  {total}")

    # Write summary
    today = datetime.now(timezone.utc).date()
    out_dir = ROOT / "logs" / "deployment" / str(today)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "align_prod_candidates_run.json"
    with open(out, "w") as f:
        json.dump({
            "run_utc": datetime.now(timezone.utc).isoformat(),
            "total_profiles": len(targets),
            "parallelism": args.parallel,
            "total_elapsed_seconds": round(time.time() - t_start, 1),
            "results": results,
        }, f, indent=2, default=str)
    print(f"\n[saved] {out}")

    # Summary counts
    ok = sum(1 for r in results if r.get("status") == "ok")
    fail = sum(1 for r in results if r.get("status") not in ("ok", None))
    print(f"\nSummary: {ok} ok, {fail} failed / errored, of {len(targets)} attempted")
    if fail > 0:
        print("Failures:")
        for r in results:
            if r.get("status") not in ("ok", None):
                print(f"  {r['profile']:<30} {r.get('status','?'):<14}  {r.get('err', r.get('tail','')[:120])}")


if __name__ == "__main__":
    main()
