"""realistic_ceiling.py -- DEPRECATED 2026-05-20.

================================================================================
DEPRECATED -- METHODOLOGY BUG -- DO NOT CITE OUTPUTS
================================================================================
This tool was built 2026-05-20 evening to provide a "realistic" ceiling at
varying hit-rates as an alternative to perfect_oracle_ceiling.py. On first run,
TRAIN compound came out at 30,999,978,273% (30 billion percent) — obviously
broken.

ROOT CAUSE:
  The Monte Carlo samples `n_winners` from the `win_5pct == 1` pool and the
  remaining K - n_winners from the `win_5pct == 0` pool. But the non-winner
  pool is NOT symmetric to the winner pool: most non-winner cells have
  max_gain_3d in (0%, +5%) — they just didn't hit clean TP — and per_pick_pnl
  clips at ±5% and deducts cost. So a "loser" pick contributes ~+0.5% to +1.5%
  on average, NOT a true loss. The simulation embeds an unstated assumption
  that the random ranker, when it misses a winner, still draws from a positively-
  drifting distribution. Daily means come out ~+1.1% at 50% hit, compounding to
  unbounded numbers over multi-year windows.

WHAT TO USE INSTEAD:
  - For a "random ranker" benchmark: N1_N3_REPORT.md (TRAIN random-K +1692% NAV,
    7d clear 50.5%) or OOS_REPLAY_REPORT.md (OOS random-K +225%, 7d clear 39%).
    These use the ACTUAL 17-setup library + ACTUAL day-by-day exits, not a
    synthetic ranker model.
  - For a "perfect ranker" benchmark: best-K results in those same reports.
  - For ranker-quality-vs-deploy comparisons: honest_v2_simulator.py with the
    4-bounds output (best/signal/random/worst-K).

DO NOT re-enable this tool without redesigning the loser-pool sampling to
match the empirical distribution of non-winner outcomes (likely: weighted
sample from a TP/SL/timeout distribution conditional on the asset's regime).
================================================================================

LEGACY DOCSTRING below (kept for provenance):

PROBLEM (2026-05-20 oracle audit):
  perfect_oracle_ceiling.py assumes 100% selection accuracy (sort by realised
  max_gain_3d). Its output is a perfect-foresight upper bound. Comparing real
  strategies against it (e.g., "82% CAPTURE_LEAK") makes ordinary performance
  look like systemic failure -- because no real ranker hits 100% precision.

THIS TOOL:
  Computes the ceiling a REALISTIC ranker can reach. Models a K=5 long-only
  spot strategy with:
    - Imperfect ranker: precision = hit_rate parameter (50%, 55%, 60%, 65%, 70%)
    - 3-day hold; +5% TP / -5% SL daily-OHLC approximation
    - 0.30% RT cost (taker bucket avg, conservative)
    - Daily compound across the full window
    - K=5 simultaneous positions cap (matches v3 paper_trade_replay current cap)

  For each hit rate, samples N_DRAWS independent simulations where each day
  randomly picks K=5 candidates from the day's eligible pool, with the
  selection biased so that {hit_rate}% of picks are TRUE wins (clean +5% TP
  before -5% stop).

OUTPUT (markdown + JSON):
  Per-hit-rate, per-split (TRAIN/VAL/OOS/UNSEEN):
    - mean daily NAV gain
    - compound NAV over window
    - annualized
    - max DD
    - Sharpe proxy
  Plus a HEADLINE delta table: how much each +5pp of hit-rate is worth.

INTERPRETATION:
  - At 50% hit (where current specialists are), realistic ceiling = $X
  - At 60% hit (achievable with good ranker), realistic ceiling = $Y
  - At 70% hit (Headline-tier), realistic ceiling = $Z
  - Strategies currently at $current_NAV map to the ceiling at their hit-rate
  - The GAP to +1%/d is mostly a hit-rate problem, not capacity

This replaces the misleading framing "we are capturing only X% of perfect
foresight" with the honest framing "at our current hit rate, we are capturing
Y% of the realistic ceiling."
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "data" / "processed" / "capturable_win_catalog.parquet"
OUT_MD = ROOT / "runs" / "audit" / "REALISTIC_CEILING_2026_05_20.md"
OUT_JSON = ROOT / "runs" / "audit" / "realistic_ceiling_2026_05_20.json"

# Sim parameters
K = 5                       # K=5 simultaneous (matches v3 current cap)
COST_RT = 0.0030            # 0.30% round-trip taker (conservative)
TP_PCT = 0.05               # +5% take profit
SL_PCT = -0.05              # -5% stop loss
N_DRAWS = 200               # independent random simulations per hit_rate
HIT_RATES = [0.50, 0.55, 0.60, 0.65, 0.70]
SIZE_PER_POS = 1.0 / K      # equal-weight K=5
TRADING_DAYS_PER_YEAR = 365


def per_pick_pnl(win_5pct: int, max_loss_3d: float, max_gain_3d: float,
                  is_picked_winner: bool) -> float:
    """Realistic per-pick net PnL under TP/SL/3d-hold daily-OHLC approximation.

    If is_picked_winner=True, the ranker correctly identified a clean +5% TP
    day; pick captures +TP_PCT - cost.

    If is_picked_winner=False, the ranker picked a non-clean-win day. Realistic
    outcome depends on actual day:
      - max_loss_3d <= -5% AND not a clean TP : -SL_PCT triggers (loss capped at -5%)
      - else : capped at min(max_gain_3d, +TP_PCT) (no full +5% achieved)
    """
    if is_picked_winner and win_5pct == 1:
        return TP_PCT - COST_RT
    # Non-winner: either stop fires or modest move
    if max_loss_3d <= -0.05:
        return SL_PCT - COST_RT
    # No stop, no TP -- capped at realised max_gain or near-zero
    return float(np.clip(max_gain_3d, -0.05, 0.05)) - COST_RT


def simulate_window(cat: pd.DataFrame, hit_rate: float, rng: np.random.Generator) -> dict:
    """Simulate K=5 daily picks across the window at the given hit rate.

    For each day:
      - Pool = all (asset, date) rows for that date
      - "Winners" pool = subset where win_5pct == 1
      - "Losers" pool = remainder
      - K_winners ~ Binomial(K, hit_rate); rest from losers
      - Each pick gets per_pick_pnl
      - Day NAV gain = mean(picks_net) (equal-weight K=5)
    """
    if cat.empty:
        return {"days": 0, "daily_mean_pct": 0.0, "comp_pct": 0.0,
                "ann_pct": 0.0, "max_dd_pct": 0.0, "sharpe": 0.0}
    daily_gains = []
    for d, g in cat.groupby("date"):
        winners = g[g["win_5pct"] == 1]
        losers = g[g["win_5pct"] == 0]
        n_winners = min(int(rng.binomial(K, hit_rate)), len(winners))
        n_losers = min(K - n_winners, len(losers))
        if n_winners + n_losers == 0:
            continue
        picks_pnl = []
        if n_winners > 0:
            wp = winners.sample(n=n_winners, random_state=rng.integers(2**31))
            for _, r in wp.iterrows():
                picks_pnl.append(per_pick_pnl(int(r["win_5pct"]),
                                                float(r["max_loss_3d"]),
                                                float(r["max_gain_3d"]),
                                                is_picked_winner=True))
        if n_losers > 0:
            lp = losers.sample(n=n_losers, random_state=rng.integers(2**31))
            for _, r in lp.iterrows():
                picks_pnl.append(per_pick_pnl(int(r["win_5pct"]),
                                                float(r["max_loss_3d"]),
                                                float(r["max_gain_3d"]),
                                                is_picked_winner=False))
        # Day NAV gain = equal-weight average across K filled picks
        # (if fewer than K filled, the remainder is cash = 0 contribution)
        day_pnl_pct = sum(picks_pnl) / K
        daily_gains.append(day_pnl_pct)
    if not daily_gains:
        return {"days": 0, "daily_mean_pct": 0.0, "comp_pct": 0.0,
                "ann_pct": 0.0, "max_dd_pct": 0.0, "sharpe": 0.0}
    arr = np.array(daily_gains)
    comp = (1 + arr).prod() - 1
    mean = arr.mean()
    std = arr.std(ddof=0)
    ann = (1 + mean) ** TRADING_DAYS_PER_YEAR - 1
    nav = (1 + arr).cumprod()
    peak = np.maximum.accumulate(nav)
    dd = (nav / peak - 1).min()
    sharpe = (mean / std * np.sqrt(TRADING_DAYS_PER_YEAR)) if std > 1e-9 else 0.0
    return {"days": int(len(arr)), "daily_mean_pct": float(mean * 100),
            "comp_pct": float(comp * 100), "ann_pct": float(ann * 100),
            "max_dd_pct": float(dd * 100), "sharpe": float(sharpe)}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("="*78)
    print("[DEPRECATED 2026-05-20] realistic_ceiling.py has a methodology bug.")
    print("DO NOT CITE OUTPUTS. See top-of-file docstring for details.")
    print("Use N1_N3_REPORT.md / OOS_REPLAY_REPORT.md / honest_v2_simulator.py instead.")
    print("="*78)
    print("Refusing to run. Exit 2.")
    return 2
    # ---- Legacy entry point kept below for reference; unreachable ----
    print("="*78)
    print("REALISTIC CEILING -- K=5 / 3d hold / TP+5/SL-5 / 0.30% cost")
    print("="*78)
    print(f"Hit-rate grid : {HIT_RATES}")
    print(f"Draws per cell: {N_DRAWS}  (Monte Carlo over random ranker)")
    print(f"K (simultaneous): {K}")
    print(f"Catalog       : {CATALOG}")

    if not CATALOG.exists():
        print(f"[FATAL] catalog missing: {CATALOG}")
        return 2
    cat = pd.read_parquet(CATALOG)
    cat["date"] = pd.to_datetime(cat["date"]).dt.normalize()
    cat = cat[cat["max_gain_3d"].notna()].copy()
    print(f"Catalog rows  : {len(cat):,}; assets {cat['asset'].nunique()}")

    SPLITS = ["train", "val", "oos", "unseen"]
    results: dict = {}
    for split in SPLITS:
        sub = cat[cat["split"] == split].copy()
        n_days = sub["date"].nunique()
        if n_days < 30:
            print(f"\n[{split}] skipped ({n_days} days)")
            continue
        print(f"\n=== Split: {split.upper()} (n={n_days} days, {len(sub):,} rows) ===")
        results[split] = {}
        # Diagnostic: per-day base rate of winners
        per_day_winrate = sub.groupby("date")["win_5pct"].mean()
        avail_pct = (sub["win_5pct"] == 1).mean() * 100
        print(f"  Base rate (any (asset,date) is a clean +5% winner): {avail_pct:.1f}%")
        print(f"  Per-day winner-availability median: {per_day_winrate.median()*100:.1f}%")
        print()
        print(f"  {'hit':>5}  {'days':>5}  {'daily_mean':>12}  {'comp_window':>13}  {'annualized':>12}  {'max_dd':>9}  {'sharpe':>8}")
        for hr in HIT_RATES:
            # Monte Carlo over N_DRAWS random rankers
            draw_metrics = []
            for seed in range(N_DRAWS):
                rng = np.random.default_rng(seed)
                m = simulate_window(sub, hr, rng)
                draw_metrics.append(m)
            # Average across draws
            agg = {k: np.mean([d[k] for d in draw_metrics])
                   for k in ("daily_mean_pct", "comp_pct", "ann_pct", "max_dd_pct", "sharpe")}
            agg["days"] = draw_metrics[0]["days"]
            agg["hit_rate"] = hr
            results[split][f"{hr:.2f}"] = agg
            print(f"  {hr:>5.2f}  {agg['days']:>5d}  {agg['daily_mean_pct']:>+11.4f}%  "
                  f"{agg['comp_pct']:>+12.2f}%  {agg['ann_pct']:>+11.2f}%  "
                  f"{agg['max_dd_pct']:>+8.2f}%  {agg['sharpe']:>+7.3f}")

    # Headline delta table: marginal value of each +5pp hit-rate step
    print("\n" + "="*78)
    print("HEADLINE -- Marginal value of each +5pp of hit-rate (compound NAV over window)")
    print("="*78)
    print(f"  {'split':<8}{'50% -> 55%':>14}{'55% -> 60%':>14}{'60% -> 65%':>14}{'65% -> 70%':>14}")
    for split in SPLITS:
        if split not in results:
            continue
        deltas = []
        for i in range(len(HIT_RATES) - 1):
            a = results[split].get(f"{HIT_RATES[i]:.2f}", {}).get("comp_pct")
            b = results[split].get(f"{HIT_RATES[i+1]:.2f}", {}).get("comp_pct")
            if a is None or b is None:
                deltas.append(None)
            else:
                deltas.append(b - a)
        ds = "  ".join(f"{(d if d is not None else 0):>+11.2f}pp" for d in deltas)
        print(f"  {split:<8}{ds}")

    # Compare to recent specialist NAVs (sourced from canonical state)
    print("\n" + "="*78)
    print("SPECIALIST MAP -- where current specialists sit on the realistic ceiling")
    print("="*78)
    print("8Q v3 paper-trade-replay results (4Q24-4Q25, 730 days):")
    print(f"  {'Specialist':<35}{'NAV 8Q':>10}{'implied_hit':>14}")
    specialists = [
        ("STRICT_LO_SETUP60 (u100)",   20.95),
        ("TA_SML_SOLO (u50)",          71.84),
        ("TA_SML_MAX_OPPS (u50)",     118.04),
        ("TA_SML_MOE (u50)",           73.31),
        ("TA_SML_MOE_ZOO (u50)",      131.29),
        ("MOVER_CONTINUATION_LO",      21.46),
    ]
    # Use OOS+UNSEEN combined ceiling as the rough 8Q-equivalent reference
    # (730 days ≈ 2 yrs of trading; OOS+UNSEEN window is ~7mo so we approx)
    ref_split = "oos" if "oos" in results else next(iter(results), None)
    if ref_split:
        # Quick implied-hit-rate lookup: find the hit rate whose comp_pct
        # over the window most closely matches each specialist's annualized
        # (scaled to the window length)
        ref = results[ref_split]
        ref_days = next(iter(ref.values()))["days"] if ref else 0
        for name, nav_8q in specialists:
            # 8Q NAV converted to annualized
            spec_ann = ((1 + nav_8q/100) ** (365/730) - 1) * 100
            # Find hit rate where ann_pct closest to spec_ann
            ann_grid = [(hr, ref[f"{hr:.2f}"]["ann_pct"]) for hr in HIT_RATES if f"{hr:.2f}" in ref]
            if not ann_grid:
                continue
            best_hr = min(ann_grid, key=lambda x: abs(x[1] - spec_ann))[0]
            print(f"  {name:<35}{nav_8q:>+9.2f}%   ~{best_hr*100:.0f}% hit")

    # Write report
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Realistic Ceiling — what's actually reachable (2026-05-20)",
        "",
        "**Method**: Monte Carlo over 200 random rankers at each hit-rate level.",
        f"Each random ranker picks K={K} candidates per day; `hit_rate%` of picks",
        f"are TRUE +5% TP winners (clean), rest are losses/modest moves under",
        f"TP=+5% / SL=-5% / 0.30% RT cost / 3d hold daily-OHLC approximation.",
        f"Daily compound across the window.",
        "",
        f"**Why this replaces the perfect-foresight ceiling**: a perfect ranker",
        f"is unreachable; the relevant benchmark is the ceiling under a realistic",
        f"ranker. This tells us **the value of each +5pp of hit-rate**.",
        "",
    ]
    for split in SPLITS:
        if split not in results:
            continue
        lines.append(f"## Split: {split.upper()}")
        lines.append("")
        lines.append("| hit_rate | days | daily mean | comp over window | annualized | max DD | Sharpe |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for hr in HIT_RATES:
            key = f"{hr:.2f}"
            if key not in results[split]:
                continue
            a = results[split][key]
            lines.append(
                f"| {hr:.2f} | {a['days']} | {a['daily_mean_pct']:+.4f}% | "
                f"{a['comp_pct']:+.2f}% | {a['ann_pct']:+.2f}% | "
                f"{a['max_dd_pct']:+.2f}% | {a['sharpe']:+.3f} |"
            )
        lines.append("")
    lines += [
        "## Marginal value of +5pp hit-rate (compound NAV over window)",
        "",
        "| split | 50→55 | 55→60 | 60→65 | 65→70 |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        if split not in results:
            continue
        ds = []
        for i in range(len(HIT_RATES) - 1):
            a = results[split].get(f"{HIT_RATES[i]:.2f}", {}).get("comp_pct")
            b = results[split].get(f"{HIT_RATES[i+1]:.2f}", {}).get("comp_pct")
            if a is None or b is None:
                ds.append("—")
            else:
                ds.append(f"{b-a:+.2f}pp")
        lines.append(f"| {split} | " + " | ".join(ds) + " |")

    lines += [
        "",
        "## Honest reading",
        "",
        "- The +1-5%/d project target = +365% to +1825% annualized.",
        "- Realistic ceiling at 70% hit-rate is the upper end of what's reachable",
        "  under daily-cadence K=5 LO with current 5-cap. Compare to project target.",
        "- Each +5pp of hit-rate is worth ~Xpp annualized; the gap from current",
        "  specialists (≈50-55% hit) to the target is mostly a **ranker-quality**",
        "  problem, not a capacity / cost / universe problem.",
        "- Closing that gap = per-asset specialist ML, conditional-on-state",
        "  filters, sub-day cadence, signature-based detection.",
        "- The 'CAPTURE_LEAK' framing in the diagnostic oracle compares strategies",
        "  to a PERFECT-FORESIGHT ceiling. That makes ordinary performance look",
        "  broken. The realistic ceiling above is the right benchmark.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    OUT_JSON.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\n[OK] wrote {OUT_MD}")
    print(f"[OK] wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
