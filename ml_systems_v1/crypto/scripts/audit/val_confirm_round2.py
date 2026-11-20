"""VAL-confirm the 3 round-2 setup additions (Supertrend x2, Ichimoku).

These were added to the 17-setup deploy portfolio based on TRAIN marginal
lift but were never validated on VAL. Per Layer-2 + anticipate-next-question
directives, deploy-tier requires VAL confirmation.

Tests each of the 3 candidates on VAL window (2023-07-02 -> 2024-05-15) and
confirms whether the setup qualifies in matching regime on VAL.
"""
from __future__ import annotations
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "oracle_layer3" / "SMART_DISCOVERY_EXHAUSTIVE_TRAIN"

VAL_START = date(2023, 7, 2)
VAL_END = date(2024, 5, 15)
COST = 0.0024
HARD_STOP = -0.04
TARGET = 0.12

# VAL gates (relaxed for shorter window)
N_MIN = 15
HIT_MIN = 0.33
ASYM_MIN = 1.2
EXPECT_MIN = 0.002
REGIMES = ("bull","chop","bear","crash")

# Reuse round-2 finders
import sys
sys.path.insert(0, str(ROOT/"scripts"/"audit"))
from extend_indicators_round2 import (
    find_supertrend, find_ichimoku, calc_supertrend, calc_ichimoku,
)

CANDIDATES = [
    ("Supertrend_flip", "(10, 2.0)", find_supertrend),
    ("Supertrend_flip", "(14, 2.5)", find_supertrend),
    ("Ichimoku_cross", "(9, 26, 52)", find_ichimoku),
]

# TRAIN qualifying regimes per setup (from round2_library.csv)
TRAIN_QUAL = {
    ("Supertrend_flip", "(10, 2.0)"): {"bull": True, "chop": True, "bear": False, "crash": False},
    ("Supertrend_flip", "(14, 2.5)"): {"bull": True, "chop": True, "bear": False, "crash": False},
    ("Ichimoku_cross", "(9, 26, 52)"): {"bull": True, "chop": True, "bear": False, "crash": True},
}

def asymmetric(rets):
    out = np.copy(rets)
    out = np.where(out <= HARD_STOP, HARD_STOP, out)
    out = np.where(out >= TARGET, TARGET, out)
    return out

def qualifies(returns):
    if len(returns) < N_MIN: return False, {}
    asym = asymmetric(returns)
    pos = returns[returns > 0]; neg = returns[returns < 0]
    ar = pos.mean()/abs(neg.mean()) if len(neg) and neg.mean() != 0 else float('inf')
    ok = (asym.mean() >= EXPECT_MIN and (returns > 0).mean() >= HIT_MIN
          and ar >= ASYM_MIN and len(returns) >= N_MIN)
    return ok, {"n": len(returns), "asym_mean_pct": asym.mean()*100,
                "hit_pct": (returns > 0).mean()*100, "asym_ratio": min(ar, 10)}

def main():
    print("="*78)
    print("VAL CONFIRMATION OF +3 ROUND-2 SETUPS")
    print("="*78)

    # Load VAL panel + regime overlay
    print("Loading VAL panel...")
    files = sorted((ROOT/"data"/"processed"/"chimera"/"1d").glob("*_v51_chimera_1d_*.parquet"))
    panels = {}
    for f in files:
        sym = f.name.split("_")[0].upper().replace("USDT","")
        try:
            df = pl.read_parquet(f, columns=["timestamp","open","high","low","close"]).to_pandas()
        except Exception: continue
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
        df = df[(df["date"] >= VAL_START - timedelta(days=120)) & (df["date"] <= VAL_END + timedelta(days=14))].reset_index(drop=True)
        if len(df) < 100: continue
        df["asset"] = sym
        panels[sym] = df
    print(f"  panels: {len(panels)} assets")

    reg = pl.read_parquet(ROOT/"runs"/"oracle_layer2"/"daily_regime_cluster.parquet").to_pandas()
    reg["date"] = pd.to_datetime(reg["date"]).dt.date
    reg = reg[(reg["date"] >= VAL_START) & (reg["date"] <= VAL_END)]
    date2reg = dict(zip(reg["date"], reg["btc_regime_30d"]))

    # Generate VAL events for each candidate
    print("\nGenerating VAL events...")
    all_rows = []
    for ind, cfg, finder in CANDIDATES:
        for asset, sub in panels.items():
            try:
                idx = finder(sub, cfg)
            except Exception: continue
            for i in idx:
                if i < 60 or i + 14 >= len(sub): continue
                ev_date = sub.iloc[i]["date"]
                if ev_date < VAL_START or ev_date > VAL_END: continue
                entry = float(sub.iloc[i]["close"])
                if entry <= 0 or not np.isfinite(entry): continue
                c14 = float(sub.iloc[i+14]["close"])
                if not np.isfinite(c14): continue
                ret = c14/entry - 1 - COST
                all_rows.append({
                    "asset": asset, "date": ev_date, "indicator": ind, "config": cfg,
                    "ret_E_14d": ret, "btc_regime_30d": date2reg.get(ev_date, "UNK"),
                })
    events = pd.DataFrame(all_rows)
    print(f"VAL events generated: {len(events):,}")
    print(events.groupby(["indicator","config"]).size())

    # Per-candidate VAL qualification
    print("\n=== VAL CONFIRMATION ===")
    results = []
    for ind, cfg, _ in CANDIDATES:
        train_qual = TRAIN_QUAL[(ind, cfg)]
        sub_ev = events[(events["indicator"]==ind) & (events["config"]==cfg)]
        val_qual = {}
        val_stats = {}
        for reg_name in REGIMES:
            r = sub_ev[sub_ev["btc_regime_30d"] == reg_name]["ret_E_14d"].dropna().values
            ok, stats = qualifies(r)
            val_qual[reg_name] = ok
            val_stats[reg_name] = stats
        # Match: TRAIN qualifies + VAL qualifies in same regime
        matches = [r for r in REGIMES if train_qual.get(r) and val_qual[r]]
        stable = len(matches) >= 1
        results.append({
            "indicator": ind, "config": cfg,
            "train_qual_regimes": ",".join(r for r in REGIMES if train_qual.get(r)),
            "val_qual_regimes": ",".join(r for r in REGIMES if val_qual[r]),
            "matched_regimes": ",".join(matches),
            "stable": stable,
            "val_stats": val_stats,
        })
        print(f"\n  {ind} {cfg}:")
        print(f"    TRAIN qualifies: {[r for r in REGIMES if train_qual.get(r)]}")
        print(f"    VAL qualifies:   {[r for r in REGIMES if val_qual[r]]}")
        for r in REGIMES:
            s = val_stats[r]
            if not s:
                print(f"      {r:<8} n<{N_MIN}; no qualification")
            else:
                print(f"      {r:<8} n={s['n']:<5} asym_mean={s['asym_mean_pct']:+.3f}%  hit={s['hit_pct']:.1f}%  asym_ratio={s['asym_ratio']:.2f}  {'OK' if val_qual[r] else 'FAIL'}")
        print(f"    -> STABLE: {stable} (matched: {matches})")

    # Final deploy set
    stable_setups = [(r["indicator"], r["config"]) for r in results if r["stable"]]
    failed_setups = [(r["indicator"], r["config"]) for r in results if not r["stable"]]
    print(f"\n=== FINAL DEPLOY VERDICT ===")
    print(f"VAL-confirmed (KEEP in deploy): {len(stable_setups)}")
    for s in stable_setups:
        print(f"  {s[0]} {s[1]}")
    print(f"VAL-failed (DROP from deploy): {len(failed_setups)}")
    for s in failed_setups:
        print(f"  {s[0]} {s[1]}")

    # Write report
    lines = ["# Round-2 VAL Confirmation\n"]
    lines.append("\n## VAL window: 2023-07-02 -> 2024-05-15\n")
    lines.append("Gates: n>=15, hit>=33%, asym>=1.2, expect>=+0.20% (slightly relaxed)\n")
    lines.append("\n## Results\n")
    lines.append("| indicator | config | TRAIN qual | VAL qual | matched | stable |")
    lines.append("|---|---|---|---|---|:--:|")
    for r in results:
        lines.append(f"| {r['indicator']} | `{r['config']}` | {r['train_qual_regimes']} | {r['val_qual_regimes']} | {r['matched_regimes']} | {'YES' if r['stable'] else 'NO'} |")

    lines.append("\n## Decision\n")
    if len(stable_setups) == 3:
        lines.append("**All 3 round-2 setups CONFIRMED on VAL. 17-setup deploy portfolio stands.**")
    elif len(stable_setups) > 0:
        lines.append(f"**{len(stable_setups)}/3 confirmed. Drop the {len(failed_setups)} failed setup(s) from deploy.**")
        lines.append(f"Final deploy size: {14 + len(stable_setups)} setups.")
    else:
        lines.append("**0/3 confirmed. Drop all 3 round-2 additions. Deploy reverts to 14-setup portfolio.**")

    (OUT_DIR/"VAL_CONFIRM_ROUND2_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    # Save deploy decision as JSON
    import json
    decision = {
        "stable_setups": stable_setups,
        "failed_setups": failed_setups,
        "final_deploy_size": 14 + len(stable_setups),
    }
    (OUT_DIR/"deploy_decision_round2.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_DIR/'VAL_CONFIRM_ROUND2_REPORT.md'}")
    print(f"Final deploy size: {decision['final_deploy_size']} setups")

if __name__ == "__main__":
    main()
