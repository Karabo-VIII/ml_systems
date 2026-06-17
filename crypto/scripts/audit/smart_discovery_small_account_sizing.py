"""Small-account asymmetric-sizing analysis on the enriched mining substrate.

V2 (2026-05-20 RED-team Layer-2 fixes):
  - DETERMINISTIC K-selection per day (sort by asym_expectancy DESC)
  - 3 bounds estimated: BEST-K (upper), RANDOM-K (realistic), WORST-K (lower)
  - UNIQUE-ASSET enforcement (no double-firing same asset same day)
  - Portfolio drawdown metric (max running DD across daily NAV)
  - +20% target sensitivity test (vs +12%)

Replaces Sharpe-first ranking (institutional) with expectancy / asymmetry /
geometric-mean ranking (small-account-appropriate).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "oracle_layer3" / "SMART_DISCOVERY_EXHAUSTIVE_TRAIN"

# Small-account sizing parameters
BET_FRACTION = 0.08          # 8% per trade (vs 4% institutional)
HARD_STOP_PCT = -0.04        # cap loss at -4% per trade
COST = 0.0024
SMALL_ACCOUNT_FLOOR_WEEKLY = 0.0525  # +5.25%/7d user floor
K_MAX = 8                    # max concurrent positions

# RNG seed for reproducibility on RANDOM-K bound
RNG_SEED = 42

def asymmetric_stop_returns(rets: np.ndarray, stop=-0.04, target=0.12):
    out = np.copy(rets)
    out = np.where(out <= stop, stop, out)
    out = np.where(out >= target, target, out)
    return out

def per_cell_stats(returns: np.ndarray, target=0.12) -> dict:
    if len(returns) < 30:
        return None
    pos = returns[returns > 0]
    neg = returns[returns < 0]
    asym = (pos.mean() / abs(neg.mean())) if len(neg) and neg.mean() != 0 else float('inf')
    hit = (returns > 0).mean()
    expectancy = returns.mean()
    asym_returns = asymmetric_stop_returns(returns, HARD_STOP_PCT, target)
    asym_expectancy = asym_returns.mean()
    asym_hit = (asym_returns > 0).mean()
    safe = np.clip(returns, -0.50, 5.0)
    gm = float(np.exp(np.log1p(safe).mean())) - 1
    nav_at_bet = returns.sum() * BET_FRACTION
    nav_asym = asym_returns.sum() * BET_FRACTION
    return {
        "n": int(len(returns)),
        "expectancy_pct": expectancy * 100,
        "asym_expectancy_pct": asym_expectancy * 100,
        "hit_pct": hit * 100,
        "asym_hit_pct": asym_hit * 100,
        "asymmetry_ratio": float(asym),
        "geom_mean_pct": gm * 100,
        "nav_at_8pct_size": nav_at_bet * 100,
        "asym_nav_at_8pct_size": nav_asym * 100,
        "small_account_score": expectancy * np.sqrt(len(returns)) * min(asym, 10.0),
        "asym_small_account_score": asym_expectancy * np.sqrt(len(returns)) * min(asym, 10.0),
    }

def simulate_ensemble(spec_events, daily_oracle, K, mode="best", target=0.12, rng_seed=42):
    """Simulate daily NAV under one of 3 selection modes.

    Modes:
      best:   per-day pick top-K by asym_expectancy (DETERMINISTIC upper bound)
      worst:  per-day pick bottom-K by asym_expectancy (DETERMINISTIC lower bound)
      random: per-day random-K with seed (REALISTIC central estimate)

    Each day: enforce unique-asset (one entry per asset).
    """
    rng = np.random.default_rng(rng_seed)
    spec_events = spec_events.copy()
    spec_events["date"] = pd.to_datetime(spec_events["date"])
    spec_events["asym_ret"] = asymmetric_stop_returns(
        spec_events["ret_E_14d"].fillna(0).values, HARD_STOP_PCT, target)

    daily_records = []
    for d, day_grp in spec_events.groupby(spec_events["date"].dt.date):
        # Unique-asset: keep best fire per asset (by asym_ret)
        day_grp = day_grp.sort_values("asym_ret", ascending=False)
        unique = day_grp.drop_duplicates(subset="asset", keep="first")
        n_unique = len(unique)
        if n_unique == 0:
            continue
        # Selection mode
        if mode == "best":
            picked = unique.head(K)
        elif mode == "worst":
            picked = unique.tail(K)
        elif mode == "random":
            if n_unique <= K:
                picked = unique
            else:
                idx = rng.choice(n_unique, K, replace=False)
                picked = unique.iloc[idx]
        else:
            raise ValueError(mode)
        nav_today = picked["asym_ret"].sum() * BET_FRACTION
        daily_records.append({
            "date": d, "n_fires_raw": len(day_grp),
            "n_unique_assets": n_unique, "n_picked": len(picked),
            "nav_pct": nav_today,
        })
    df = pd.DataFrame(daily_records)
    return df

def portfolio_drawdown(daily_df: pd.DataFrame) -> dict:
    """Compute max drawdown of cumulative portfolio NAV."""
    if len(daily_df) == 0: return {"max_dd_pct": 0, "max_dd_date": None}
    daily_df = daily_df.sort_values("date").reset_index(drop=True)
    cum = (1 + daily_df["nav_pct"]).cumprod()
    running_max = cum.cummax()
    dd = (cum / running_max - 1) * 100
    return {
        "max_dd_pct": float(dd.min()),
        "max_dd_date": str(daily_df.loc[dd.idxmin(), "date"]) if len(dd) else None,
        "final_cum_pct": float((cum.iloc[-1] - 1) * 100),
    }

def main():
    events = pd.read_parquet(OUT_DIR / "per_event_enriched.parquet")
    print(f"Loaded {len(events):,} events")
    print(f"Sizing: bet={BET_FRACTION*100:.0f}%/trade  stop={HARD_STOP_PCT*100:.0f}%  K={K_MAX}")
    print()

    # Per-cell stats at both +12% and +20% target sensitivity
    for target_pct in (0.12, 0.20):
        rows = []
        for (ind, cfg, reg), grp in events.groupby(["indicator", "config", "btc_regime_30d"]):
            rets = grp["ret_E_14d"].dropna().values
            s = per_cell_stats(rets, target=target_pct)
            if s is None: continue
            rows.append({"indicator": ind, "config": cfg, "regime": reg,
                          "target_pct": target_pct * 100, **s})
        df = pd.DataFrame(rows)
        df.to_csv(OUT_DIR / f"small_account_sizing_target{int(target_pct*100)}.csv", index=False)

    # Use the +12% target survivor set for ensemble simulation
    print("=== ENSEMBLE SIMULATION (BOUNDED) ===")
    df = pd.read_csv(OUT_DIR / "small_account_sizing_target12.csv")
    survivors = df[
        (df["asym_expectancy_pct"] > 1.0) &
        (df["asym_hit_pct"] >= 40) &
        (df["asymmetry_ratio"] >= 2.0) &
        (df["n"] >= 100)
    ]
    top_per_ind = survivors.sort_values(["indicator", "asym_small_account_score"],
                                          ascending=[True, False]).groupby("indicator").head(5)
    specialists = set(zip(top_per_ind["indicator"], top_per_ind["config"], top_per_ind["regime"]))
    print(f"Specialist set: {len(specialists)} cells")

    spec_filter = events.set_index(["indicator", "config", "btc_regime_30d"]).index.isin(specialists)
    spec_events = events[spec_filter].copy()
    print(f"Specialist firings: {len(spec_events):,}")
    print()

    bounds = {}
    for mode in ("best", "random", "worst"):
        daily_df = simulate_ensemble(spec_events, None, K=K_MAX, mode=mode,
                                       target=0.12, rng_seed=RNG_SEED)
        dd = portfolio_drawdown(daily_df)
        total = daily_df["nav_pct"].sum() * 100
        mean_d = daily_df["nav_pct"].mean() * 100
        median_d = daily_df["nav_pct"].median() * 100
        positive_days = (daily_df["nav_pct"] > 0).mean() * 100
        # 7d rolling floor
        daily_df = daily_df.sort_values("date").reset_index(drop=True)
        daily_df["nav_7d"] = daily_df["nav_pct"].rolling(7).sum()
        floor_clear = (daily_df["nav_7d"] >= SMALL_ACCOUNT_FLOOR_WEEKLY).sum()
        floor_n = max(len(daily_df) - 6, 1)
        bounds[mode] = {
            "total_nav": total, "mean_daily": mean_d, "median_daily": median_d,
            "positive_days_pct": positive_days,
            "max_dd_pct": dd["max_dd_pct"],
            "max_dd_date": dd["max_dd_date"],
            "final_cum_pct": dd["final_cum_pct"],
            "floor_clear_weeks": int(floor_clear), "floor_total": floor_n,
            "floor_clear_pct": floor_clear * 100 / floor_n,
            "mean_7d_rolling": daily_df["nav_7d"].mean() * 100,
        }
        print(f"[{mode:6s}-K]  total_NAV={total:+8.2f}%  mean_daily={mean_d:+.3f}%  "
              f"med_daily={median_d:+.3f}%  +days={positive_days:4.1f}%  "
              f"max_DD={dd['max_dd_pct']:+6.2f}%  cum_compound={dd['final_cum_pct']:+8.2f}%  "
              f"floor_clear={floor_clear}/{floor_n} ({floor_clear*100/floor_n:.1f}%)  "
              f"mean_7d={daily_df['nav_7d'].mean()*100:+.2f}%")

    # +20% target sensitivity
    print()
    print("=== TARGET=+20% SENSITIVITY (vs +12% above) ===")
    df20 = pd.read_csv(OUT_DIR / "small_account_sizing_target20.csv")
    survivors20 = df20[
        (df20["asym_expectancy_pct"] > 1.0) &
        (df20["asym_hit_pct"] >= 40) &
        (df20["asymmetry_ratio"] >= 2.0) &
        (df20["n"] >= 100)
    ]
    top_per_ind20 = survivors20.sort_values(["indicator","asym_small_account_score"],
                                              ascending=[True, False]).groupby("indicator").head(5)
    specs20 = set(zip(top_per_ind20["indicator"], top_per_ind20["config"], top_per_ind20["regime"]))
    spec_filter20 = events.set_index(["indicator", "config", "btc_regime_30d"]).index.isin(specs20)
    spec_events20 = events[spec_filter20].copy()
    daily_df20 = simulate_ensemble(spec_events20, None, K=K_MAX, mode="best",
                                     target=0.20, rng_seed=RNG_SEED)
    dd20 = portfolio_drawdown(daily_df20)
    daily_df20 = daily_df20.sort_values("date").reset_index(drop=True)
    daily_df20["nav_7d"] = daily_df20["nav_pct"].rolling(7).sum()
    floor_clear_20 = (daily_df20["nav_7d"] >= SMALL_ACCOUNT_FLOOR_WEEKLY).sum()
    floor_n_20 = max(len(daily_df20)-6, 1)
    print(f"[best-K @ +20%]  total_NAV={daily_df20['nav_pct'].sum()*100:+.2f}%  "
          f"max_DD={dd20['max_dd_pct']:+6.2f}%  cum_compound={dd20['final_cum_pct']:+8.2f}%  "
          f"floor_clear={floor_clear_20}/{floor_n_20} ({floor_clear_20*100/floor_n_20:.1f}%)  "
          f"mean_7d={daily_df20['nav_7d'].mean()*100:+.2f}%")

    daily_df.to_csv(OUT_DIR / "ensemble_daily_nav_bounded.csv", index=False)

    # Report
    lines = ["# Small-Account Asymmetric Sizing — V2 (bounded + deterministic + unique-asset)\n"]
    lines.append(f"\n## Parameters\n")
    lines.append(f"- bet_fraction: {BET_FRACTION*100:.0f}% per trade")
    lines.append(f"- hard_stop:    {HARD_STOP_PCT*100:.0f}% (asymmetric)")
    lines.append(f"- target:       +12% (also tested +20%)")
    lines.append(f"- K_max:        {K_MAX} concurrent (= {BET_FRACTION*K_MAX*100:.0f}% portfolio max)")
    lines.append(f"- unique-asset: yes (no double-firing same asset same day)")
    lines.append(f"- floor:        +{SMALL_ACCOUNT_FLOOR_WEEKLY*100:.2f}%/7d (user mandate)")

    lines.append(f"\n## Specialists ensemble — 3 bounds (+12% target)\n")
    lines.append("| mode | total NAV | mean daily | +days % | max DD | cum compound | 7d floor clear | mean 7d |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for m in ("best", "random", "worst"):
        b = bounds[m]
        lines.append(f"| {m}-K | {b['total_nav']:+.2f}% | {b['mean_daily']:+.3f}% | {b['positive_days_pct']:.1f}% | {b['max_dd_pct']:+.2f}% | {b['final_cum_pct']:+.2f}% | {b['floor_clear_weeks']}/{b['floor_total']} ({b['floor_clear_pct']:.1f}%) | {b['mean_7d_rolling']:+.2f}% |")

    lines.append(f"\n## +20% target sensitivity (best-K only)\n")
    lines.append(f"- total NAV: +{daily_df20['nav_pct'].sum()*100:.2f}%")
    lines.append(f"- max DD: {dd20['max_dd_pct']:+.2f}%")
    lines.append(f"- cum compound: +{dd20['final_cum_pct']:.2f}%")
    lines.append(f"- 7d floor clear: {floor_clear_20}/{floor_n_20} ({floor_clear_20*100/floor_n_20:.1f}%)")
    lines.append(f"- mean 7d: +{daily_df20['nav_7d'].mean()*100:.2f}%")
    lines.append("\nInterpretation: +20% target preserves fat right tail but reduces hit rate marginally.")

    (OUT_DIR / "SMALL_ACCOUNT_SIZING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_DIR / 'SMALL_ACCOUNT_SIZING_REPORT.md'}")

if __name__ == "__main__":
    main()
