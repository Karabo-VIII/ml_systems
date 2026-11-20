"""retrain_t4.py — quarterly v51_full_T4 ranker retrain.

Per PROD_DEPLOY_VERDICT_2026_05_09 mandate: T4 ranker rolls every quarter
on a fresh 12-15 month training window with 3-month purge before the
test/deploy window. Next due: 2026-08-01.

Re-uses the panel-build + training logic from
scripts/strat_audit/production_backtest_2026_jan_apr.py:build_t4_seed,
but exposes:
  --cutoff YYYY-MM-DD   Training data cutoff (TRAIN_END override)
  --test-start YYYY-MM-DD  Test window start (default = cutoff + 3-month purge)
  --test-end YYYY-MM-DD    Test window end (default = today - 1 day)
  --out-pickle PATH        Where to write the new pickle (default versioned)
  --out-seed-dir PATH      Where to write the daily_snapshot.csv (default versioned)
  --force                  Rebuild even if pickle exists

Usage:
    python scripts/retrain_t4.py --cutoff 2026-08-01

Schedule:
  - 2026-08-01 (Q3 retrain) — first scheduled retrain after live deploy
  - 2026-11-01 (Q4)
  - 2027-02-01 (Q1)
  - ... etc.

LOUD failure on missing chimera v51 panels or insufficient training rows.
"""
from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "strat_audit"))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Quarterly v51_full_T4 ranker retrain")
    ap.add_argument("--cutoff", required=True,
                    help="TRAIN_END date YYYY-MM-DD")
    ap.add_argument("--test-start", default=None,
                    help="TEST_START. Default: cutoff + 3-month purge.")
    ap.add_argument("--test-end", default=None,
                    help="TEST_END. Default: today - 1 day.")
    ap.add_argument("--out-pickle", default=None,
                    help="Pickle path. Default: models/xsec_ranker/xgb_ndcg_v1_u87_v51full_T4_<cutoff>.pkl")
    ap.add_argument("--out-seed-dir", default=None,
                    help="Seed dir. Default: logs/paper_trader_v2/seeds/pt_xsec_v51full_T4_<cutoff>/")
    ap.add_argument("--force", action="store_true",
                    help="Rebuild even if pickle exists")
    args = ap.parse_args()

    import pandas as pd
    cutoff_ts = pd.Timestamp(args.cutoff)
    test_start = pd.Timestamp(args.test_start) if args.test_start \
        else cutoff_ts + pd.DateOffset(months=3)
    test_end = pd.Timestamp(args.test_end) if args.test_end \
        else pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    cutoff_tag = cutoff_ts.strftime("%Y%m%d")

    out_pickle = Path(args.out_pickle) if args.out_pickle \
        else ROOT / "models" / "xsec_ranker" / f"xgb_ndcg_v1_u87_v51full_T4_{cutoff_tag}.pkl"
    out_seed_dir = Path(args.out_seed_dir) if args.out_seed_dir \
        else ROOT / "logs" / "paper_trader_v2" / "seeds" / f"pt_xsec_v51full_T4_{cutoff_tag}"

    if out_pickle.exists() and not args.force:
        print(f"[t4] pickle exists at {out_pickle.relative_to(ROOT)}; pass --force to rebuild")
        return 0

    print(f"[t4] cutoff={cutoff_ts.date()} test_window={test_start.date()}..{test_end.date()}")
    print(f"[t4] out_pickle={out_pickle.relative_to(ROOT)}")
    print(f"[t4] out_seed={out_seed_dir.relative_to(ROOT)}")

    # Override the constants in the production_backtest script then call its
    # build_t4_seed (single source of truth for T4 build logic).
    import production_backtest_2026_jan_apr as pb
    pb.TRAIN_END = cutoff_ts.strftime("%Y-%m-%d")
    pb.TEST_START = test_start.strftime("%Y-%m-%d")
    pb.TEST_END = test_end.strftime("%Y-%m-%d")
    pb.T4_SEED_DIR = out_seed_dir

    # build_t4_seed() inside production_backtest_2026_jan_apr writes pickle to
    # a HARDCODED default path. To produce a versioned pickle without
    # clobbering the existing default (which the live deploy points at), we:
    #   1) back up the existing default pickle (if present)
    #   2) call build_t4_seed (which writes to default path)
    #   3) move the new pickle to the versioned path
    #   4) restore the backup to the default path
    # Failure at any step rolls back the default pickle.
    import shutil
    orig_build = pb.build_t4_seed
    target_pickle = out_pickle
    default_pickle = ROOT / "models" / "xsec_ranker" / "xgb_ndcg_v1_u87_v51full_T4.pkl"
    backup_pickle = default_pickle.with_suffix(".pkl.bak")

    def patched_build():
        if out_seed_dir.exists() and not args.force:
            print(f"[t4] seed dir exists at {out_seed_dir.relative_to(ROOT)}; --force required")
            return
        # Step 1: backup default pickle if present and target is versioned
        backed_up = False
        if default_pickle.exists() and target_pickle != default_pickle:
            shutil.copy2(default_pickle, backup_pickle)
            backed_up = True
            print(f"[t4] backed up existing default pickle -> {backup_pickle.name}")
        try:
            out_seed_dir.mkdir(parents=True, exist_ok=True)
            # Step 2: run the original build (writes to default path)
            orig_build()
            # Step 3: move new pickle to versioned path
            if target_pickle != default_pickle and default_pickle.exists():
                shutil.move(str(default_pickle), str(target_pickle))
                print(f"[t4] versioned pickle -> {target_pickle.relative_to(ROOT)}")
            # Step 4: restore the backup
            if backed_up and backup_pickle.exists():
                shutil.move(str(backup_pickle), str(default_pickle))
                print(f"[t4] restored prior default pickle (live deploy path unchanged)")
        except Exception:
            # Rollback: if a backup exists, restore it
            if backed_up and backup_pickle.exists():
                shutil.move(str(backup_pickle), str(default_pickle))
                print(f"[t4] error during retrain; rolled back default pickle")
            raise

    patched_build()
    print(f"[t4] retrain complete")
    print(f"[t4] next steps:")
    print(f"     1) Update config/production_blends.yaml ranker field for the v51_full_T4 sleeve")
    print(f"        from xgb_ndcg_v1_u87_v51full_T4 -> xgb_ndcg_v1_u87_v51full_T4_{cutoff_tag}")
    print(f"     2) Update CONSERVATIVE/PRIME blend seed from")
    print(f"        pt_xsec_v51full_T4_2026_05_05 -> pt_xsec_v51full_T4_{cutoff_tag}")
    print(f"     3) Run python -m src.strategy.gen5_growth.blend_composer --check to verify wiring")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
