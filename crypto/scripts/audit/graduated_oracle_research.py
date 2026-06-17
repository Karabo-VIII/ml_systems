"""graduated_oracle_research.py — Graduated-oracle precision-tier ceiling analysis.

QUESTION: which oracle precision-tier (100%/90%/80%/...) can our deployed
sleeve mimic? Where is the discoverable ceiling under our exit framework?

METHOD:
  For each tier p ∈ {1.0, 0.9, ..., 0.0}:
    - Each day, rank universe by 5-day forward return (oracle signal aligned
      with our typical exit horizon).
    - Per open slot: with prob p use the oracle's top pick; else uniform-random
      from the remaining universe.
    - Run through SAME portfolio mechanics as deployed sleeve:
        K=8, per-asset 10%, total 60%, exits: hard -4% / trail-arm +5% / trail-drop -3% / hold 14d.
    - N_SAMPLES Monte Carlo runs per tier (smooth random component).

  Find which tier our deployed +91.40% mimics → that is our effective
  selection-precision target.

ACCOUNTING (MtM, daily):
  Each day d, for each held position:
    - nav_change += weight * (close_d - last_close) / last_close   [overnight move]
    - update max_profit_pct, bars_held, last_close
    - exit-check (only fires when bars_held >= 1)
  Then entries: open at close_d with last_close=close_d, bars_held=0.

OUTPUT:
  runs/audit/MA_EMA_PROFILE_2026_05_20/GRADUATED_ORACLE_RESEARCH.md
  runs/audit/MA_EMA_PROFILE_2026_05_20/graduated_oracle_per_tier.csv
  runs/audit/MA_EMA_PROFILE_2026_05_20/graduated_oracle_daily.csv
"""
from __future__ import annotations

import sys
import glob
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
CHIMERA_1D = ROOT / "data" / "processed" / "chimera" / "1d"
OUT_DIR = ROOT / "runs" / "audit" / "MA_EMA_PROFILE_2026_05_20"
OUT_MD = OUT_DIR / "GRADUATED_ORACLE_RESEARCH.md"
OUT_PER_TIER = OUT_DIR / "graduated_oracle_per_tier.csv"
OUT_DAILY = OUT_DIR / "graduated_oracle_daily.csv"
STRAT_DAILY = OUT_DIR / "oos_union_daily.csv"

OOS_START = _date(2024, 5, 16)
OOS_END = _date(2025, 3, 15)

K_MAX = 8
PER_ASSET_CAP = 0.10
TOTAL_DEPLOY_CAP = 0.60
HARD_STOP = -0.04
TRAIL_ARM_PROFIT = 0.05
TRAIL_DROP = 0.03
HOLD_BARS = 14
ORACLE_FWD_BARS = 5  # forward-window for oracle's ranking signal

N_SAMPLES = 30
TIERS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
SEED = 42


def load_universe():
    print("Loading chimera 1d closes...")
    asset_close = {}
    for f in glob.glob(str(CHIMERA_1D / "*_v51_chimera_1d_*.parquet")):
        sym = Path(f).name.split("_")[0].upper().replace("USDT", "")
        try:
            df = pl.read_parquet(f, columns=["timestamp", "close"]).to_pandas()
        except Exception:
            continue
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.normalize()
        df = df.sort_values("date").drop_duplicates("date")
        if len(df) < 30:
            continue
        asset_close[sym] = df.set_index("date")["close"]
    print(f"  {len(asset_close)} assets loaded")
    return asset_close


def build_close_matrix(asset_close):
    """Build wide DataFrame of close prices indexed by date."""
    return pd.DataFrame(asset_close)


def simulate_tier(close_mat, dates, precision, rng):
    """Run one MC simulation at precision p. Returns (daily_df, trades_df)."""
    nav = 1.0
    open_positions = {}  # asset -> dict
    daily = []
    trades = []

    # Pre-compute forward returns matrix for oracle signal
    # fwd_ret[t][asset] = (close[t+ORACLE_FWD_BARS] - close[t]) / close[t]
    close_arr = close_mat.values  # (T, N)
    assets = list(close_mat.columns)
    T = len(close_mat)
    fwd_ret = np.full((T, len(assets)), np.nan)
    for i in range(T - ORACLE_FWD_BARS):
        c0 = close_arr[i]
        c1 = close_arr[i + ORACLE_FWD_BARS]
        with np.errstate(invalid="ignore", divide="ignore"):
            fwd_ret[i] = (c1 - c0) / c0

    # Day-index lookup
    date_to_idx = {d: i for i, d in enumerate(close_mat.index)}

    for d in dates:
        t = date_to_idx.get(pd.Timestamp(d))
        if t is None:
            continue
        close_today = close_arr[t]  # (N,)

        # ----- Phase A: NAV accrual + exit checks for existing positions -----
        nav_change = 0.0
        new_open = {}
        for sym, pos in open_positions.items():
            j = assets.index(sym)  # asset index
            ct = close_today[j]
            if not np.isfinite(ct):
                # Asset missing today — force flush at last_close, no PnL change
                trades.append({
                    "asset": sym,
                    "entry_date": pos["entry_date"],
                    "exit_date": d,
                    "weight": pos["weight"],
                    "ret_pct": ((pos["last_close"] - pos["entry_price"]) / pos["entry_price"]) * 100,
                    "bars_held": pos["bars_held"],
                    "exit_reason": "MISSING",
                })
                continue

            # Overnight accrual
            day_ret = (ct - pos["last_close"]) / pos["last_close"]
            nav_change += pos["weight"] * day_ret

            # Update state
            ret_from_entry = (ct - pos["entry_price"]) / pos["entry_price"]
            pos["max_profit_pct"] = max(pos["max_profit_pct"], ret_from_entry)
            pos["bars_held"] += 1
            pos["last_close"] = ct

            # Exit decision (only when bars_held >= 1)
            exit_reason = None
            if ret_from_entry <= HARD_STOP:
                exit_reason = "HARD_STOP"
            elif pos["max_profit_pct"] >= TRAIL_ARM_PROFIT:
                drop_from_peak = pos["max_profit_pct"] - ret_from_entry
                if drop_from_peak >= TRAIL_DROP:
                    exit_reason = "TRAIL"
            if exit_reason is None and pos["bars_held"] >= HOLD_BARS:
                exit_reason = "TIME_STOP"

            if exit_reason:
                trades.append({
                    "asset": sym,
                    "entry_date": pos["entry_date"],
                    "exit_date": d,
                    "weight": pos["weight"],
                    "ret_pct": ret_from_entry * 100,
                    "bars_held": pos["bars_held"],
                    "exit_reason": exit_reason,
                })
            else:
                new_open[sym] = pos

        nav *= (1.0 + nav_change)
        open_positions = new_open

        # ----- Phase B: New entries at today's close -----
        deployed = sum(p["weight"] for p in open_positions.values())
        slots_open = K_MAX - len(open_positions)
        room_for_new = max(0.0, TOTAL_DEPLOY_CAP - deployed)

        if slots_open > 0 and room_for_new >= PER_ASSET_CAP - 1e-9:
            # Find available assets with finite close and finite oracle signal
            fwd = fwd_ret[t]  # (N,)
            avail_mask = np.isfinite(close_today) & np.isfinite(fwd)
            for sym in open_positions:
                j = assets.index(sym)
                avail_mask[j] = False
            avail_idx = np.where(avail_mask)[0]

            if len(avail_idx) > 0:
                # Rank by forward return descending
                ranked_idx = avail_idx[np.argsort(-fwd[avail_idx])]
                oracle_picks = ranked_idx[:slots_open]

                # Per-slot precision draw
                picks = []
                used = set()
                # First: deterministic oracle picks for some slots
                for i in range(slots_open):
                    if room_for_new < PER_ASSET_CAP - 1e-9:
                        break
                    if rng.random() < precision and i < len(oracle_picks):
                        pick_idx = int(oracle_picks[i])
                        if pick_idx in used:
                            continue
                    else:
                        # Random pick from unused
                        pool = [k for k in avail_idx if k not in used]
                        if not pool:
                            break
                        pick_idx = int(rng.choice(pool))
                    used.add(pick_idx)
                    picks.append(pick_idx)
                    w = min(PER_ASSET_CAP, room_for_new)
                    sym = assets[pick_idx]
                    ct = float(close_today[pick_idx])
                    open_positions[sym] = {
                        "entry_date": d,
                        "entry_price": ct,
                        "last_close": ct,
                        "weight": w,
                        "max_profit_pct": 0.0,
                        "bars_held": 0,
                    }
                    room_for_new -= w

        # ----- Record day -----
        daily.append({
            "date": d,
            "nav": nav,
            "n_open": len(open_positions),
            "deployed_pct": sum(p["weight"] for p in open_positions.values()) * 100,
        })

    # Flush at window end
    last_d = dates[-1]
    t = date_to_idx.get(pd.Timestamp(last_d))
    if t is not None:
        close_last = close_arr[t]
        for sym, pos in open_positions.items():
            j = assets.index(sym)
            ct = close_last[j] if np.isfinite(close_last[j]) else pos["last_close"]
            ret = (ct - pos["entry_price"]) / pos["entry_price"]
            trades.append({
                "asset": sym,
                "entry_date": pos["entry_date"],
                "exit_date": last_d,
                "weight": pos["weight"],
                "ret_pct": ret * 100,
                "bars_held": pos["bars_held"],
                "exit_reason": "FLUSH",
            })

    return pd.DataFrame(daily), pd.DataFrame(trades)


def compute_metrics(daily_df, trades_df):
    if daily_df.empty:
        return {"nav_pct": 0, "ann_pct": 0, "daily_pct": 0, "sortino": 0, "max_dd": 0, "trades": 0}
    nav_final = float(daily_df["nav"].iloc[-1])
    nav_pct = (nav_final - 1.0) * 100
    n_days = len(daily_df)
    daily_ret = daily_df["nav"].pct_change().dropna()
    mean_d = float(daily_ret.mean() * 100)
    ann = (nav_final ** (365 / max(n_days, 1)) - 1) * 100
    downside = daily_ret[daily_ret < 0]
    if len(downside) > 1 and downside.std() > 0:
        sortino = float((daily_ret.mean() / downside.std()) * np.sqrt(365))
    else:
        sortino = 0.0
    cummax = daily_df["nav"].cummax()
    dd = (daily_df["nav"] / cummax - 1) * 100
    max_dd = float(dd.min())
    return {
        "nav_pct": nav_pct,
        "ann_pct": ann,
        "daily_pct": mean_d,
        "sortino": sortino,
        "max_dd": max_dd,
        "trades": len(trades_df),
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=" * 78)
    print("GRADUATED ORACLE PRECISION-TIER RESEARCH")
    print("=" * 78)
    print(f"Window: {OOS_START} -> {OOS_END}")
    print(f"Tiers: {TIERS}")
    print(f"N_SAMPLES per tier: {N_SAMPLES}")
    print(f"K={K_MAX} / per-asset {PER_ASSET_CAP*100}% / total {TOTAL_DEPLOY_CAP*100}%")
    print(f"Exits: hard {HARD_STOP*100}% / trail+{TRAIL_ARM_PROFIT*100}%-{TRAIL_DROP*100}% / hold {HOLD_BARS}d")
    print(f"Oracle signal: forward {ORACLE_FWD_BARS}-bar return")
    print()

    asset_close = load_universe()
    close_mat = build_close_matrix(asset_close)
    # Restrict to OOS window with small forward buffer
    full_start = pd.Timestamp(OOS_START)
    full_end = pd.Timestamp(OOS_END) + pd.Timedelta(days=ORACLE_FWD_BARS + 1)
    close_mat = close_mat.loc[full_start:full_end].sort_index()
    print(f"Close matrix shape: {close_mat.shape}")

    dates = [d.date() for d in close_mat.index if OOS_START <= d.date() <= OOS_END]
    print(f"OOS days: {len(dates)}")

    tier_metrics = []
    tier_curves = {}

    for tier in TIERS:
        print(f"\n--- TIER precision={tier:.2f} ---")
        nav_curves = []
        metric_rows = []
        for s in range(N_SAMPLES):
            rng = np.random.default_rng(SEED + s + int(tier * 1000))
            d_df, t_df = simulate_tier(close_mat, dates, tier, rng)
            m = compute_metrics(d_df, t_df)
            metric_rows.append(m)
            nav_curves.append(d_df.set_index("date")["nav"])
        nav_mat = pd.concat(nav_curves, axis=1)
        nav_mat.columns = [f"s{i}" for i in range(len(nav_curves))]
        mean_curve = nav_mat.mean(axis=1)
        tier_curves[tier] = mean_curve

        dfm = pd.DataFrame(metric_rows)
        agg = {
            "tier": tier,
            "nav_pct_mean": float(dfm["nav_pct"].mean()),
            "nav_pct_std": float(dfm["nav_pct"].std()),
            "nav_pct_p25": float(dfm["nav_pct"].quantile(0.25)),
            "nav_pct_p75": float(dfm["nav_pct"].quantile(0.75)),
            "ann_pct_mean": float(dfm["ann_pct"].mean()),
            "daily_pct_mean": float(dfm["daily_pct"].mean()),
            "sortino_mean": float(dfm["sortino"].mean()),
            "max_dd_mean": float(dfm["max_dd"].mean()),
            "trades_mean": float(dfm["trades"].mean()),
        }
        tier_metrics.append(agg)
        print(f"  NAV: {agg['nav_pct_mean']:+.2f}% +/- {agg['nav_pct_std']:.2f}% "
              f"(p25={agg['nav_pct_p25']:+.2f}, p75={agg['nav_pct_p75']:+.2f})")
        print(f"  Sortino: {agg['sortino_mean']:+.2f}  DD: {agg['max_dd_mean']:+.2f}%  "
              f"trades: {agg['trades_mean']:.0f}")

    tier_df = pd.DataFrame(tier_metrics)
    tier_df.to_csv(OUT_PER_TIER, index=False)
    print(f"\n[OK] wrote {OUT_PER_TIER}")

    curves_df = pd.DataFrame(tier_curves)
    curves_df.index.name = "date"
    curves_df.to_csv(OUT_DAILY)
    print(f"[OK] wrote {OUT_DAILY}")

    # Compare to strat
    strat = pd.read_csv(STRAT_DAILY)
    strat["date"] = pd.to_datetime(strat["date"])
    strat = strat.set_index("date").sort_index()
    strat_nav = float(strat["portfolio_value"].iloc[-1])
    strat_pct = (strat_nav - 1.0) * 100
    print(f"\nDeployed UNION sleeve final NAV: +{strat_pct:.2f}%")

    # Implied precision via linear interpolation on tier curve
    sorted_tiers = tier_df.sort_values("tier")
    above = sorted_tiers[sorted_tiers["nav_pct_mean"] >= strat_pct]
    below = sorted_tiers[sorted_tiers["nav_pct_mean"] < strat_pct]
    if not above.empty and not below.empty:
        upper = above.iloc[above["nav_pct_mean"].argmin()]
        lower = below.iloc[below["nav_pct_mean"].argmax()]
        if upper["nav_pct_mean"] != lower["nav_pct_mean"]:
            frac = (strat_pct - lower["nav_pct_mean"]) / (upper["nav_pct_mean"] - lower["nav_pct_mean"])
            implied = float(lower["tier"] + frac * (upper["tier"] - lower["tier"]))
        else:
            implied = float(lower["tier"])
    else:
        implied = None
    if implied is not None:
        print(f"Implied effective precision: {implied:.3f} ({implied*100:.1f}%)")

    # Write markdown
    lines = [
        "# Graduated-Oracle Precision-Tier Research (2026-05-20)\n",
        f"**Window**: {OOS_START} -> {OOS_END} ({len(dates)} days)",
        f"**Constraints**: K={K_MAX}, per-asset {PER_ASSET_CAP*100:.0f}%, total {TOTAL_DEPLOY_CAP*100:.0f}%",
        f"**Exits**: hard {HARD_STOP*100:.0f}% / trail arm +{TRAIL_ARM_PROFIT*100:.0f}% drop -{TRAIL_DROP*100:.0f}% / hold {HOLD_BARS} days",
        f"**Oracle signal**: forward {ORACLE_FWD_BARS}-bar return (next-week-typical)",
        f"**MC samples**: {N_SAMPLES} per tier",
        "",
        "## Headline",
        "",
        f"- Deployed UNION sleeve OOS NAV: **+{strat_pct:.2f}%**",
    ]
    if implied is not None:
        lines.append(f"- **Effective precision (vs perfect-K=8 forward picker): {implied:.3f} "
                     f"({implied*100:.1f}%)**")
    lines.extend([
        "",
        "## Precision-tier ladder (MC means)",
        "",
        "| Tier (p) | NAV % | p25 % | p75 % | Daily % | Sortino | Max DD % | Trades |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for _, r in tier_df.iterrows():
        lines.append(
            f"| {r['tier']:.2f} | {r['nav_pct_mean']:+.2f} | {r['nav_pct_p25']:+.2f} | "
            f"{r['nav_pct_p75']:+.2f} | {r['daily_pct_mean']:+.4f} | "
            f"{r['sortino_mean']:+.2f} | {r['max_dd_mean']:+.2f} | {r['trades_mean']:.0f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"- **Tier 1.00** (perfect-K=8 forward picker) = +{tier_df.iloc[0]['nav_pct_mean']:.2f}% — "
        f"theoretical ceiling under our exit framework + constraints.",
        f"- **Tier 0.00** (pure random) = {tier_df.iloc[-1]['nav_pct_mean']:+.2f}% — exit-framework-only "
        f"noise floor.",
        f"- **Deployed sleeve** at +{strat_pct:.2f}% sits at implied precision "
        f"{f'{implied:.2f} ({implied*100:.0f}%)' if implied else 'OUT_OF_RANGE'}.",
        "",
        "## What this tells us about the gap",
        "",
        "- The exit framework's own contribution is the gap between tier 0.00 (random) and 1.00 (perfect).",
        "- Selection skill = how high we climb that ladder.",
        "- A 5pp shift in precision typically moves NAV by a measurable amount — see the per-tier deltas.",
        "",
        "## Realistic deploy targets",
        "",
        "- **Current**: ~{}% precision, +{:.0f}% NAV/10mo".format(
            f"{implied*100:.0f}" if implied else "?", strat_pct),
        "- **Achievable +1 tier (+10pp precision)**: see tier table for expected lift.",
        "- **Achievable +2 tier (+20pp precision)**: similarly bounded.",
        "",
        "## Caveats",
        "",
        "- Look-ahead in oracle selection is INTENTIONAL — defines the ceiling.",
        "- Exit mechanics use only bar-by-bar info from entry onward (no look-ahead).",
        "- 1d bars only — sub-day cadence ceiling not measured here.",
        f"- Oracle signal uses {ORACLE_FWD_BARS}-bar forward return (aligns with typical exit horizon, "
        "not 1d which would overweight one-day pops).",
        "",
    ])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {OUT_MD}")


if __name__ == "__main__":
    main()
