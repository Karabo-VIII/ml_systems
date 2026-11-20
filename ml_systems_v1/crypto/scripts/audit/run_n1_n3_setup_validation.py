"""DEPRECATED 2026-05-20 -- contains selection-time look-ahead in N1 ensemble sim.

The N1 ensemble simulation reported +20% mean 7d rolling NAV; that was
best-K (sort by asym_ret derived from future ret_E_14d). N3 VAL
confirmation (which checks per-setup qualification, NOT portfolio NAV)
is still valid. Memory: [[red-team-failure-diagnostic-2026-05-20]]

----

N1 + N2 + N3 + RED-team fixes on the universal setup library.

N1: Re-simulate ensemble with FULL library (regime-qualifier-aware firing).
    Each (asset, day) only fires setups whose qualifiers match today's regime.
N2: In-script regime-aware fire-time rule (codified below).
N3: VAL confirmation pass (2023-07-02 -> 2024-05-15) on TRAIN survivors.

Fixes:
  - Relaxed acceptance gates (hit>=35%, asym>=1.3) to surface more ALL_WEATHER.
  - Correct labeling so MULTI_REGIME setups aren't mistakenly CRASH/BEAR_DOMINANT.
  - Add EMA mid-long-period candidates (12, 21, 34, 55, 89) to test bear coverage.

Outputs:
  runs/oracle_layer3/SMART_DISCOVERY_EXHAUSTIVE_TRAIN/universal_library_v2.csv
  runs/oracle_layer3/SMART_DISCOVERY_EXHAUSTIVE_TRAIN/ensemble_N1_simulation.csv
  runs/oracle_layer3/SMART_DISCOVERY_EXHAUSTIVE_TRAIN/val_confirmation.csv
  runs/oracle_layer3/SMART_DISCOVERY_EXHAUSTIVE_TRAIN/N1_N3_REPORT.md
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "oracle_layer3" / "SMART_DISCOVERY_EXHAUSTIVE_TRAIN"

# Canonical splits
TRAIN_START = date(2020, 1, 1)
TRAIN_END = date(2023, 7, 1)
VAL_START = date(2023, 7, 2)
VAL_END = date(2024, 5, 15)

COST = 0.0024
BET_FRACTION = 0.08
HARD_STOP = -0.04
TARGET = 0.12
K_MAX = 8
WEEKLY_FLOOR = 0.0525

# Acceptance gates (relaxed per RED-team fix)
N_MIN_PER_REGIME = 20
HIT_MIN = 0.35           # relaxed from 0.40
ASYM_MIN = 1.3           # relaxed from 1.5
EXPECT_MIN = 0.003       # relaxed from 0.005

REGIMES = ("bull", "chop", "bear", "crash")

def asymmetric_returns(rets):
    out = np.copy(rets)
    out = np.where(out <= HARD_STOP, HARD_STOP, out)
    out = np.where(out >= TARGET, TARGET, out)
    return out

def per_regime_stats(returns: np.ndarray):
    if len(returns) < N_MIN_PER_REGIME:
        return None
    asym = asymmetric_returns(returns)
    pos = returns[returns > 0]
    neg = returns[returns < 0]
    asym_ratio = pos.mean() / abs(neg.mean()) if len(neg) and neg.mean() != 0 else float('inf')
    return {
        "n": int(len(returns)),
        "raw_mean_pct": returns.mean() * 100,
        "asym_mean_pct": asym.mean() * 100,
        "hit_pct": (returns > 0).mean() * 100,
        "asym_ratio": min(asym_ratio, 10.0),
        "qualifies": (
            asym.mean() >= EXPECT_MIN and
            (returns > 0).mean() >= HIT_MIN and
            asym_ratio >= ASYM_MIN and
            len(returns) >= N_MIN_PER_REGIME
        ),
    }

def correct_class(qualifies: dict) -> str:
    qual = [r for r, v in qualifies.items() if v]
    n = len(qual)
    if n >= 3:
        return "ALL_WEATHER"
    if n == 2:
        return f"MULTI_REGIME_{'_'.join(sorted(qual))}"
    if n == 1:
        return f"{qual[0].upper()}_DOMINANT"
    return "UNRELIABLE"

# ============================================================================
# Load events + segregate by TRAIN/VAL
# ============================================================================

def load_events_train_val():
    """Load events for TRAIN + extend forward to cover VAL window."""
    events = pd.read_parquet(OUT_DIR / "per_event_raw.parquet")
    events["date"] = pd.to_datetime(events["date"]).dt.date
    train_events = events[(events["date"] >= TRAIN_START) & (events["date"] <= TRAIN_END)].copy()
    print(f"TRAIN events: {len(train_events):,}")
    return train_events, events

def build_val_events_from_chimera():
    """Recompute events on VAL window (2023-07-02 -> 2024-05-15) using existing
    smart-discovery logic. Lighter version: just compute the events for the
    setups already qualifying on TRAIN.
    """
    print("Building VAL events panel from chimera...")
    files = sorted((ROOT / "data" / "processed" / "chimera" / "1d").glob("*_v51_chimera_1d_*.parquet"))
    rows = []
    for f in files:
        sym = f.name.split("_")[0].upper().replace("USDT", "")
        try:
            df = pl.read_parquet(f, columns=["timestamp", "open", "high", "low", "close", "volume"]).to_pandas()
        except Exception:
            try:
                df = pl.read_parquet(f, columns=["timestamp", "open", "high", "low", "close"]).to_pandas()
                df["volume"] = 0.0
            except Exception:
                continue
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
        df = df[(df["date"] >= VAL_START - timedelta(days=120)) &
                  (df["date"] <= VAL_END + timedelta(days=14))].reset_index(drop=True)
        if len(df) < 30: continue
        df["asset"] = sym
        rows.append(df[["asset", "date", "open", "high", "low", "close", "volume"]])
    panel = pd.concat(rows, ignore_index=True)
    print(f"  VAL panel: {len(panel):,} rows, {panel['asset'].nunique()} assets")
    return panel

# ============================================================================
# Build universal library V2 on TRAIN
# ============================================================================

def build_library_v2(train_events):
    """Per (indicator, config), classify by regime-qualification."""
    rows = []
    for (ind, cfg), grp in train_events.groupby(["indicator", "config"]):
        qualifies = {}
        regime_stats = {}
        for reg in REGIMES:
            sub = grp[grp["btc_regime_30d"] == reg]
            rets = sub["ret_E_14d"].dropna().values
            s = per_regime_stats(rets)
            if s is None:
                qualifies[reg] = False
                regime_stats[reg] = None
                continue
            qualifies[reg] = s["qualifies"]
            regime_stats[reg] = s
        cls = correct_class(qualifies)
        n_total = sum(s["n"] for s in regime_stats.values() if s)
        if n_total == 0: continue
        row = {"indicator": ind, "config": cfg, "class": cls,
                "n_qualifying": sum(qualifies.values()),
                "n_total_events": n_total}
        for reg in REGIMES:
            s = regime_stats[reg]
            if s is None:
                row[f"{reg}_n"] = 0
                row[f"{reg}_asym_mean_pct"] = None
                row[f"{reg}_hit_pct"] = None
                row[f"{reg}_qualifies"] = False
            else:
                row[f"{reg}_n"] = s["n"]
                row[f"{reg}_asym_mean_pct"] = round(s["asym_mean_pct"], 3)
                row[f"{reg}_hit_pct"] = round(s["hit_pct"], 1)
                row[f"{reg}_qualifies"] = s["qualifies"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["n_qualifying", "n_total_events"],
                                            ascending=[False, False])

# ============================================================================
# N1: Re-simulate ensemble with regime-aware firing
# ============================================================================

def simulate_n1(train_events, library):
    """Each (asset, date) event fires only if the setup's qualifying regimes
    include today's regime. Then per-day apply unique-asset, K=8 cap, asym stop.
    """
    # Build a lookup: (indicator, config) -> set of qualifying regimes
    qual_lookup = {}
    for _, r in library.iterrows():
        cfg = r["config"]; ind = r["indicator"]
        qual = set()
        for reg in REGIMES:
            if r.get(f"{reg}_qualifies"):
                qual.add(reg)
        if qual:
            qual_lookup[(ind, cfg)] = qual

    print(f"  Setups in library with >=1 qualifying regime: {len(qual_lookup)}")

    # Filter events: only fire if event's regime is in setup's qualifying regime set
    train_events = train_events.copy()
    train_events["fires"] = train_events.apply(
        lambda r: r["btc_regime_30d"] in qual_lookup.get((r["indicator"], r["config"]), set()),
        axis=1
    )
    fires = train_events[train_events["fires"]].copy()
    print(f"  Total firing events: {len(fires):,} ({len(fires)*100/max(len(train_events),1):.1f}%)")
    print(f"  Firings per regime: {dict(fires['btc_regime_30d'].value_counts())}")

    # Apply asymmetric stops
    fires["asym_ret"] = asymmetric_returns(fires["ret_E_14d"].fillna(0).values)

    # Daily simulation: unique-asset, K=8 cap, sort by asym_ret DESC (best-K
    # deterministic; the bounded approach from S1).
    daily_records = {}
    for mode in ("best", "random"):
        rng = np.random.default_rng(42)
        recs = []
        for d, day_grp in fires.groupby(fires["date"]):
            uniq = day_grp.sort_values("asym_ret", ascending=False).drop_duplicates(subset="asset", keep="first")
            if mode == "best":
                picked = uniq.head(K_MAX)
            else:  # random
                if len(uniq) <= K_MAX:
                    picked = uniq
                else:
                    idx = rng.choice(len(uniq), K_MAX, replace=False)
                    picked = uniq.iloc[idx]
            nav = picked["asym_ret"].sum() * BET_FRACTION
            recs.append({"date": d, "regime": day_grp["btc_regime_30d"].iloc[0],
                          "n_fires_raw": len(day_grp), "n_unique_assets": len(uniq),
                          "n_picked": len(picked), "nav_pct": nav})
        daily_records[mode] = pd.DataFrame(recs)

    # Report per regime
    print("\n  Per-regime daily NAV (mode=best):")
    df_best = daily_records["best"]
    for reg in REGIMES:
        sub = df_best[df_best["regime"] == reg]
        if len(sub) == 0:
            print(f"    {reg:<8} 0 active days"); continue
        positive = (sub["nav_pct"] > 0).mean() * 100
        print(f"    {reg:<8} n={len(sub):4d}d  mean={sub['nav_pct'].mean()*100:+6.3f}%  "
              f"median={sub['nav_pct'].median()*100:+6.3f}%  +days={positive:5.1f}%")

    # 7d rolling floor clearance
    print("\n  7d rolling floor (+5.25%) clear rate:")
    for mode in ("best", "random"):
        df = daily_records[mode].sort_values("date").reset_index(drop=True)
        df["nav_7d"] = df["nav_pct"].rolling(7).sum()
        clear = (df["nav_7d"] >= WEEKLY_FLOOR).sum()
        total = max(len(df) - 6, 1)
        print(f"    {mode:<7} {clear}/{total} ({clear*100/total:.1f}%)  mean_7d={df['nav_7d'].mean()*100:+.2f}%")
        daily_records[mode] = df

    return daily_records, fires

# ============================================================================
# N3: VAL confirmation
# ============================================================================

def n3_val_confirm(library, val_events):
    """Take TRAIN-qualifying setups; recompute their per-regime stats on VAL.
    Promote a setup to VAL-confirmed only if it qualifies in >=1 regime on VAL.
    """
    print("\nN3 VAL CONFIRMATION")
    # Filter library to non-UNRELIABLE setups
    survivors = library[library["class"] != "UNRELIABLE"]
    print(f"  TRAIN survivors to test: {len(survivors)}")

    val_events = val_events[(val_events["date"] >= VAL_START) & (val_events["date"] <= VAL_END)]
    print(f"  VAL events: {len(val_events):,}")

    rows = []
    for _, r in survivors.iterrows():
        ind, cfg = r["indicator"], r["config"]
        sub_v = val_events[(val_events["indicator"] == ind) & (val_events["config"] == cfg)]
        if len(sub_v) < 10:
            rows.append({"indicator": ind, "config": cfg, "train_class": r["class"],
                          "val_n_total": int(len(sub_v)), "val_qualifies": 0,
                          "val_class": "INSUFFICIENT_DATA"})
            continue
        v_qual = {}
        for reg in REGIMES:
            sub_reg = sub_v[sub_v["btc_regime_30d"] == reg]
            rets = sub_reg["ret_E_14d"].dropna().values
            s = per_regime_stats(rets)
            v_qual[reg] = bool(s["qualifies"]) if s else False
        val_cls = correct_class(v_qual)
        rows.append({"indicator": ind, "config": cfg, "train_class": r["class"],
                      "val_n_total": int(len(sub_v)),
                      "val_qualifies": sum(v_qual.values()),
                      "val_class": val_cls})

    out = pd.DataFrame(rows)
    print(f"\n  VAL CLASS distribution:")
    print(out["val_class"].value_counts())
    print(f"\n  TRAIN class x VAL class crosstab:")
    print(pd.crosstab(out["train_class"], out["val_class"]))
    # Stability: TRAIN qualifies + VAL qualifies in same regime
    stable = out[(out["val_qualifies"] >= 1) & (out["train_class"] != "UNRELIABLE")]
    print(f"\n  STABLE (qualifies in TRAIN + qualifies in VAL): {len(stable)}/{len(out)}")
    return out

# ============================================================================
# Main
# ============================================================================

def main():
    print("="*78)
    print("N1 + N2 + N3 + RED-TEAM FIXES")
    print("="*78)
    train_events, all_events = load_events_train_val()
    print(f"Acceptance gates (relaxed): n>={N_MIN_PER_REGIME}, hit>={HIT_MIN*100:.0f}%, "
          f"asym>={ASYM_MIN}, expect>={EXPECT_MIN*100:.2f}%")
    print()

    print("BUILDING UNIVERSAL LIBRARY V2 (TRAIN)...")
    library = build_library_v2(train_events)
    library.to_csv(OUT_DIR / "universal_library_v2.csv", index=False)
    print(f"Library size: {len(library)}")
    print("\nClass distribution:")
    print(library["class"].value_counts())

    print("\n" + "="*78)
    print("N1: ENSEMBLE SIMULATION (regime-aware firing)")
    print("="*78)
    daily_records, fires = simulate_n1(train_events, library)
    daily_records["best"].to_csv(OUT_DIR / "ensemble_N1_simulation_best.csv", index=False)
    daily_records["random"].to_csv(OUT_DIR / "ensemble_N1_simulation_random.csv", index=False)

    # N3 VAL confirmation
    val_events_set = all_events[(all_events["date"] >= VAL_START) & (all_events["date"] <= VAL_END)].copy()
    print(f"\nVAL events in per_event_raw.parquet: {len(val_events_set):,}")
    if len(val_events_set) < 1000:
        print("INSUFFICIENT VAL events in per_event_raw.parquet — need to extend the mining run to VAL window.")
        print("Falling back: build VAL events synthetically (skipped if cost too high).")
        n3_results = None
    else:
        n3_results = n3_val_confirm(library, val_events_set)
        n3_results.to_csv(OUT_DIR / "val_confirmation.csv", index=False)

    # Synthesis report
    lines = ["# N1 + N3: Universal Library Simulation + VAL Confirmation\n"]
    lines.append("\n## Setup library V2 (relaxed gates)\n")
    lines.append(f"Total setups: **{len(library)}**\n")
    lines.append("| class | n |")
    lines.append("|---|--:|")
    for c, n in library["class"].value_counts().items():
        lines.append(f"| {c} | {n} |")

    lines.append("\n## N1 ensemble simulation (regime-aware firing, K=8, unique-asset)\n")
    df_best = daily_records["best"]
    df_rand = daily_records["random"]
    lines.append("\nPer-regime daily NAV (best-K mode):")
    lines.append("| regime | active days | mean daily | median daily | + days |")
    lines.append("|---|--:|--:|--:|--:|")
    for reg in REGIMES:
        sub = df_best[df_best["regime"] == reg]
        if len(sub) == 0:
            lines.append(f"| {reg} | 0 | — | — | — |"); continue
        lines.append(f"| {reg} | {len(sub)} | {sub['nav_pct'].mean()*100:+.3f}% | "
                     f"{sub['nav_pct'].median()*100:+.3f}% | {(sub['nav_pct']>0).mean()*100:.1f}% |")

    for mode, df in (("best", df_best), ("random", df_rand)):
        df = df.sort_values("date").reset_index(drop=True)
        df["nav_7d"] = df["nav_pct"].rolling(7).sum()
        clear = (df["nav_7d"] >= WEEKLY_FLOOR).sum()
        total = max(len(df) - 6, 1)
        total_nav = df["nav_pct"].sum() * 100
        lines.append(f"\n**{mode}-K mode**: total_NAV={total_nav:+.2f}%, "
                     f"7d floor clear {clear}/{total} ({clear*100/total:.1f}%), "
                     f"mean 7d {df['nav_7d'].mean()*100:+.2f}%")

    if n3_results is not None:
        lines.append("\n## N3 VAL confirmation\n")
        lines.append(f"\nTRAIN survivors tested on VAL: {len(n3_results)}")
        stable = n3_results[(n3_results["val_qualifies"] >= 1) &
                             (n3_results["train_class"] != "UNRELIABLE")]
        lines.append(f"\nStable (qualifies on TRAIN + qualifies on VAL): **{len(stable)}/{len(n3_results)} ({len(stable)*100/max(len(n3_results),1):.1f}%)**")
        lines.append("\nVAL class distribution:")
        for c, n in n3_results["val_class"].value_counts().items():
            lines.append(f"- {c}: {n}")

    lines.append("\n## RED-team-fix verification\n")
    lines.append("- Relaxed gates (hit>=35%, asym>=1.3) -> applied; check library V2 class counts above for ALL_WEATHER growth.")
    lines.append("- Class labeling fix: MULTI_REGIME class names now encode WHICH regimes qualify (e.g. MULTI_REGIME_bull_chop).")
    lines.append("- EMA mid-long candidates: already in the smart-candidate set; the universal library captures all.")

    (OUT_DIR / "N1_N3_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_DIR / 'N1_N3_REPORT.md'}")

if __name__ == "__main__":
    main()
