"""Paranoid validation of asymmetric strategies + 4-sleeve blend.

Tests run:
    A1. Shuffle-entry control for asym_breakout   (perm entries, compare Sharpe)
    A2. Shuffle-entry control for asym_vol_expansion
    A3. Exit-order bug check: optimistic (trail-first) vs conservative (low-first)
        In optimistic, same-day trail update happens BEFORE low check. If price
        went low-first on that day, real exit would have been at OLD stop, not
        NEW higher stop. Measure magnitude of bias.
    A4. Independent verification of 4-sleeve aggregator: recompute Sharpe/DD
        from raw seed CSVs and compare to aggregator output.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import glob
import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
SEEDS_DIR = ROOT / "logs" / "paper_trader_v2" / "seeds"

sys.path.insert(0, str(ROOT / "src" / "strategy"))
from universe import UNIVERSE_50_LIQUID
UNIVERSE = set(UNIVERSE_50_LIQUID)

MAKER_RT = 0.08
CAPITAL = 10000.0
TEST_START = "2025-01-01"
TEST_END = "2026-04-22"


def build_panel():
    rows = []
    for fp in sorted(glob.glob(str(DATA / "*_chimera.parquet"))):
        asset = Path(fp).stem.replace("usdt_v50_chimera", "").upper()
        if asset not in UNIVERSE:
            continue
        df = pl.read_parquet(fp, columns=["timestamp", "close", "high", "low", "open"]).to_pandas()
        if len(df) < 1000: continue
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
        d = df.groupby("date").agg({"close": "last", "high": "max", "low": "min", "open": "first"}).reset_index()
        d["ret_d"] = d["close"].pct_change()
        d["vol_1d"] = d["ret_d"].abs()
        d["vol_z30"] = (d["vol_1d"] - d["ret_d"].rolling(30).std()) / (d["ret_d"].rolling(30).std() + 1e-9)
        d["high_prev_10d"] = d["high"].shift(1).rolling(10).max()
        d["asset"] = asset
        rows.append(d)
    return pd.concat(rows, ignore_index=True).dropna(subset=["ret_d"]).assign(
        date=lambda x: pd.to_datetime(x["date"])).sort_values(["asset", "date"]).reset_index(drop=True)


def _sim_core(panel, entry_fn, exit_order="trail_first",
              max_concurrent=10, pct_per_trade=0.10, reentry_cooldown=2,
              shuffle_entry_dates=False, shuffle_seed=42):
    """Generic portfolio runner with optional shuffle + selectable exit order.

    exit_order:
      'trail_first' (current): update trail with today's high, then check low (optimistic bias)
      'low_first'   (conservative): check low vs yesterday's stop FIRST, then update trail
    """
    test = panel[(panel["date"] >= TEST_START) & (panel["date"] <= TEST_END)]
    all_dates = sorted(test["date"].unique())
    lookups = {d: grp for d, grp in test.groupby("date")}

    # Build entry index: which (asset, date) pairs produce a valid entry signal
    entry_hits = []
    for idx, row in test.iterrows():
        spec = entry_fn(row)
        if spec is not None:
            entry_hits.append((row["date"], row["asset"], row["close"], spec, spec.get("strength", 0)))

    if shuffle_entry_dates and len(entry_hits):
        # Shuffle the entry DATES while keeping the (asset, spec) pairings
        rng = np.random.RandomState(shuffle_seed)
        shuffled_dates = rng.permutation([d for d, _, _, _, _ in entry_hits])
        entry_hits = [(shuffled_dates[i], asset, None, spec, strength)
                       for i, (_, asset, _, spec, strength) in enumerate(entry_hits)]

    # Group entries by date
    entries_by_date = {}
    for d, asset, close, spec, strength in entry_hits:
        entries_by_date.setdefault(d, []).append((asset, close, spec, strength))

    cash = CAPITAL
    positions = {}
    last_exit = {}
    trades = []
    daily_eq = []

    for d in all_dates:
        lkup = lookups.get(d)
        if lkup is None: continue
        close_map = dict(zip(lkup["asset"], lkup["close"]))
        low_map = dict(zip(lkup["asset"], lkup["low"]))
        high_map = dict(zip(lkup["asset"], lkup["high"]))

        # --- Exits ---
        closed = []
        for asset, pos in list(positions.items()):
            if asset not in close_map: continue

            if exit_order == "low_first":
                # 1) check if low breaches YESTERDAY's stop (stored in pos["stop"])
                exit_price, reason = None, None
                if low_map[asset] <= pos["stop"]:
                    exit_price = pos["stop"]; reason = "stop"
                elif (d - pos["entry_date"]).days >= pos["max_hold"]:
                    exit_price = close_map[asset]; reason = "time"
                # 2) if survived, update trail for tomorrow
                if exit_price is None:
                    if high_map[asset] > pos["peak"]:
                        pos["peak"] = high_map[asset]
                        if pos.get("trail_pct"):
                            new_trail = pos["peak"] * (1 - pos["trail_pct"])
                            if new_trail > pos["stop"]:
                                pos["stop"] = new_trail
            else:  # trail_first (current code)
                # 1) Update trail with today's high
                if high_map[asset] > pos["peak"]:
                    pos["peak"] = high_map[asset]
                    if pos.get("trail_pct"):
                        new_trail = pos["peak"] * (1 - pos["trail_pct"])
                        if new_trail > pos["stop"]:
                            pos["stop"] = new_trail
                # 2) Check low vs updated stop
                exit_price, reason = None, None
                if low_map[asset] <= pos["stop"]:
                    exit_price = pos["stop"]; reason = "stop"
                elif (d - pos["entry_date"]).days >= pos["max_hold"]:
                    exit_price = close_map[asset]; reason = "time"

            if exit_price is not None:
                size = pos["size"]
                pnl = size * (exit_price / pos["entry_price"] - 1)
                pnl -= size * (MAKER_RT / 200.0)
                cash += size + pnl
                net = (exit_price / pos["entry_price"] - 1) * 100 - MAKER_RT
                trades.append({
                    "asset": asset, "entry_date": pos["entry_date"], "exit_date": d,
                    "net_ret_pct": net, "exit_reason": reason,
                })
                closed.append(asset); last_exit[asset] = d
        for a in closed: del positions[a]

        # --- Entries ---
        if len(positions) < max_concurrent:
            today_entries = entries_by_date.get(d, [])
            # Need to (possibly) refresh close price if shuffled
            cands = []
            for asset, close, spec, strength in today_entries:
                if asset in positions: continue
                if asset in last_exit and (d - last_exit[asset]).days < reentry_cooldown:
                    continue
                # If shuffled, use the spec date's close (already in spec) OR today's close.
                # For simplicity: use today's close (matches what real-time entry would do)
                actual_close = close_map.get(asset)
                if actual_close is None: continue
                cands.append((strength, asset, actual_close, spec))
            cands.sort(reverse=True)
            for strength, asset, close, spec in cands[:max_concurrent - len(positions)]:
                size = cash * pct_per_trade
                if size > cash: break
                cash -= size + size * (MAKER_RT / 200.0)
                # Re-scale stop based on new entry price
                trail_pct = spec.get("trail_pct", 0)
                positions[asset] = {
                    "entry_date": d, "entry_price": close, "peak": close,
                    "stop": close * (1 - (spec.get("init_stop", 0.02))),
                    "trail_pct": trail_pct, "max_hold": spec.get("max_hold", 15),
                    "size": size,
                }

        pos_val = sum(p["size"] * (close_map.get(a, p["entry_price"]) / p["entry_price"])
                      for a, p in positions.items() if a in close_map)
        daily_eq.append({"date": d, "equity": cash + pos_val, "n_positions": len(positions)})

    return pd.DataFrame(daily_eq), pd.DataFrame(trades)


def breakout_entry(row):
    bh = row.get("high_prev_10d")
    if pd.isna(bh) or row["close"] <= bh:
        return None
    return {"init_stop": 0.02, "trail_pct": 0.05, "max_hold": 15,
            "strength": (row["close"] - bh) / bh}


def vol_exp_entry(row):
    if pd.isna(row.get("vol_z30")) or pd.isna(row.get("ret_d")): return None
    if row["vol_z30"] < 2.0: return None
    if row["ret_d"] < -0.02: return None
    return {"init_stop": 0.02, "trail_pct": 0.05, "max_hold": 5,
            "strength": row["vol_z30"]}


def stats(eq_df, trades_df):
    if len(eq_df) < 2: return {"status": "insufficient"}
    eq = eq_df["equity"].values
    total = (eq[-1] / CAPITAL - 1) * 100
    days = (eq_df["date"].iloc[-1] - eq_df["date"].iloc[0]).days or 1
    cagr = ((eq[-1] / CAPITAL) ** (365 / days) - 1) * 100
    dr = np.diff(eq) / eq[:-1]
    sharpe = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
    dd = ((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min() * 100
    return {"cagr_pct": cagr, "sharpe": sharpe, "max_dd_pct": dd,
            "total_ret_pct": total, "n_trades": len(trades_df)}


print("[panel] building...")
t0 = time.time()
panel = build_panel()
print(f"[panel] {panel.shape} in {time.time()-t0:.1f}s")


# =============================================================================
# A1: Shuffle-entry control on asym_breakout
# =============================================================================
print("\n" + "=" * 80)
print("A1. SHUFFLE-ENTRY CONTROL: asym_breakout")
print("=" * 80)
print("If real alpha, shuffling entry dates should drop Sharpe significantly.")
print(f"{'variant':<35} {'Sharpe':>7} {'CAGR%':>8} {'DD%':>7} {'n_tr':>5}")
print("-" * 70)

real_eq, real_tr = _sim_core(panel, breakout_entry, exit_order="trail_first")
m_real = stats(real_eq, real_tr)
print(f"{'REAL breakout':<35} {m_real['sharpe']:>+6.2f} {m_real['cagr_pct']:>+7.2f} "
      f"{m_real['max_dd_pct']:>+6.2f} {m_real['n_trades']:>5}")

shuf_results_A1 = []
for seed in [42, 1, 7, 100, 2026]:
    s_eq, s_tr = _sim_core(panel, breakout_entry, exit_order="trail_first",
                            shuffle_entry_dates=True, shuffle_seed=seed)
    m = stats(s_eq, s_tr)
    m["seed"] = seed
    shuf_results_A1.append(m)
    print(f"{'SHUFFLED breakout seed=' + str(seed):<35} {m['sharpe']:>+6.2f} "
          f"{m['cagr_pct']:>+7.2f} {m['max_dd_pct']:>+6.2f} {m['n_trades']:>5}")

mean_shuf_sharpe = np.mean([r["sharpe"] for r in shuf_results_A1])
mean_shuf_cagr = np.mean([r["cagr_pct"] for r in shuf_results_A1])
print(f"\n  REAL Sharpe:    {m_real['sharpe']:+.2f}  CAGR: {m_real['cagr_pct']:+.2f}%")
print(f"  SHUFFLED mean:  {mean_shuf_sharpe:+.2f}  CAGR: {mean_shuf_cagr:+.2f}%")
if m_real['sharpe'] > mean_shuf_sharpe + 1.0:
    print(f"  [PASS] Real breakout Sharpe > shuffled by {m_real['sharpe'] - mean_shuf_sharpe:.2f} -- signal is REAL")
else:
    print(f"  [FAIL] Real <= shuffled+1 -- signal may be noise")


# =============================================================================
# A2: Shuffle-entry control on asym_vol_expansion
# =============================================================================
print("\n" + "=" * 80)
print("A2. SHUFFLE-ENTRY CONTROL: asym_vol_expansion")
print("=" * 80)
print(f"{'variant':<35} {'Sharpe':>7} {'CAGR%':>8} {'DD%':>7} {'n_tr':>5}")
print("-" * 70)

real_eq_B, real_tr_B = _sim_core(panel, vol_exp_entry, exit_order="trail_first")
m_real_B = stats(real_eq_B, real_tr_B)
print(f"{'REAL vol_expansion':<35} {m_real_B['sharpe']:>+6.2f} "
      f"{m_real_B['cagr_pct']:>+7.2f} {m_real_B['max_dd_pct']:>+6.2f} {m_real_B['n_trades']:>5}")

shuf_results_A2 = []
for seed in [42, 1, 7, 100, 2026]:
    s_eq, s_tr = _sim_core(panel, vol_exp_entry, exit_order="trail_first",
                            shuffle_entry_dates=True, shuffle_seed=seed)
    m = stats(s_eq, s_tr)
    m["seed"] = seed
    shuf_results_A2.append(m)
    print(f"{'SHUFFLED vol_exp seed=' + str(seed):<35} {m['sharpe']:>+6.2f} "
          f"{m['cagr_pct']:>+7.2f} {m['max_dd_pct']:>+6.2f} {m['n_trades']:>5}")

mean_B_shuf_sh = np.mean([r["sharpe"] for r in shuf_results_A2])
mean_B_shuf_cg = np.mean([r["cagr_pct"] for r in shuf_results_A2])
print(f"\n  REAL Sharpe:    {m_real_B['sharpe']:+.2f}  CAGR: {m_real_B['cagr_pct']:+.2f}%")
print(f"  SHUFFLED mean:  {mean_B_shuf_sh:+.2f}  CAGR: {mean_B_shuf_cg:+.2f}%")
if m_real_B['sharpe'] > mean_B_shuf_sh + 1.0:
    print(f"  [PASS] Real vol_exp Sharpe > shuffled by {m_real_B['sharpe'] - mean_B_shuf_sh:.2f} -- REAL")
else:
    print(f"  [FAIL] Real <= shuffled+1 -- may be noise")


# =============================================================================
# A3: Exit-order bug check
# =============================================================================
print("\n" + "=" * 80)
print("A3. EXIT-ORDER BUG CHECK (optimistic vs conservative)")
print("=" * 80)
print("If low-first (conservative) gives meaningfully worse Sharpe/CAGR than")
print("trail-first (current), our prior numbers are biased optimistic.")
print(f"{'variant':<40} {'Sharpe':>7} {'CAGR%':>8} {'DD%':>7}")
print("-" * 70)

for name, fn in [("breakout", breakout_entry), ("vol_expansion", vol_exp_entry)]:
    opt_eq, opt_tr = _sim_core(panel, fn, exit_order="trail_first")
    m_opt = stats(opt_eq, opt_tr)
    con_eq, con_tr = _sim_core(panel, fn, exit_order="low_first")
    m_con = stats(con_eq, con_tr)
    print(f"{name + ' (optimistic, trail-first)':<40} {m_opt['sharpe']:>+6.2f} "
          f"{m_opt['cagr_pct']:>+7.2f} {m_opt['max_dd_pct']:>+6.2f}")
    print(f"{name + ' (conservative, low-first)':<40} {m_con['sharpe']:>+6.2f} "
          f"{m_con['cagr_pct']:>+7.2f} {m_con['max_dd_pct']:>+6.2f}")
    diff_sharpe = m_opt['sharpe'] - m_con['sharpe']
    diff_cagr = m_opt['cagr_pct'] - m_con['cagr_pct']
    if abs(diff_sharpe) < 0.2 and abs(diff_cagr) < 10:
        print(f"  [OK] bias <0.2 Sharpe / <10pp CAGR -- exit-order race is minor\n")
    else:
        print(f"  [WARN] bias Sharpe {diff_sharpe:+.2f} / CAGR {diff_cagr:+.2f}pp -- meaningful optimistic bias\n")


# =============================================================================
# A4: Independent aggregator verification
# =============================================================================
print("\n" + "=" * 80)
print("A4. INDEPENDENT AGGREGATOR VERIFICATION: recommended_4sleeve_alpha_stack")
print("=" * 80)

blend_spec = [
    ("xsec_K10_10_FULL_dneut_U50",  "pt_xsec_K10_10_FULL_dneut_U50",  0.40),
    ("frontier_dib_flow_both",      "pt_frontier_dib_flow_both",       0.25),
    ("asym_breakout",                "pt_asym_breakout",                 0.20),
    ("asym_vol_expansion",           "pt_asym_vol_expansion",            0.15),
]

# Read each sleeve's daily_snapshot.csv
sleeve_series = {}
for profile, seed, w in blend_spec:
    fp = SEEDS_DIR / seed / "daily_snapshot.csv"
    df = pd.read_csv(fp)
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "total_equity"]].sort_values("date").reset_index(drop=True)
    df["eq_normalized"] = df["total_equity"] / df["total_equity"].iloc[0]
    sleeve_series[profile] = df
    print(f"  {profile:<30} n_days={len(df)} start={df['date'].iloc[0].date()} end={df['date'].iloc[-1].date()}")

# Intersect dates
common_dates = None
for p, df in sleeve_series.items():
    ds = set(df["date"].dt.date)
    common_dates = ds if common_dates is None else common_dates & ds
common = sorted(common_dates)
print(f"\n  Common dates: {len(common)}  ({common[0]} to {common[-1]})")

# Weighted sum of normalized equities
total_w = sum(w for _, _, w in blend_spec)
portfolio_eq = np.zeros(len(common))
for profile, _, w in blend_spec:
    df = sleeve_series[profile]
    df["date_d"] = df["date"].dt.date
    df2 = pd.DataFrame({"date_d": common}).merge(df, on="date_d", how="left")
    df2["eq_normalized"] = df2["eq_normalized"].ffill().bfill()
    portfolio_eq += (w / total_w) * df2["eq_normalized"].values
portfolio_eq *= CAPITAL

# Metrics
total = (portfolio_eq[-1] / CAPITAL - 1) * 100
days = (common[-1] - common[0]).days or 1
cagr = ((portfolio_eq[-1] / CAPITAL) ** (365 / days) - 1) * 100
dr = np.diff(portfolio_eq) / portfolio_eq[:-1]
sharpe = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
dd = ((portfolio_eq - np.maximum.accumulate(portfolio_eq)) / np.maximum.accumulate(portfolio_eq)).min() * 100
sortino_dr = dr[dr < 0]
sortino = (dr.mean() / sortino_dr.std() * np.sqrt(365)) if len(sortino_dr) > 0 and sortino_dr.std() > 0 else 0

print(f"\n  Independent recompute:")
print(f"    CAGR:    {cagr:+.2f}%  Sharpe: {sharpe:+.2f}  Sortino: {sortino:+.2f}  DD: {dd:+.2f}%")
print(f"  Aggregator reported (from prior log):")
print(f"    CAGR: +91.12%  Sharpe: +6.48  Sortino: +23.70  DD: -1.79%")
# Diff
if abs(cagr - 91.12) < 1 and abs(sharpe - 6.48) < 0.2:
    print(f"  [PASS] Independent verification matches aggregator output")
else:
    print(f"  [DIFF] Sharpe diff {sharpe-6.48:+.2f} CAGR diff {cagr-91.12:+.2f}pp")


# Save
from datetime import datetime, timezone
out_dir = ROOT / "logs" / "deployment" / str(datetime.now(timezone.utc).date())
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "paranoid_validation.json"
with open(out, "w") as f:
    json.dump({
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "A1_breakout_real": m_real,
        "A1_breakout_shuffled": shuf_results_A1,
        "A2_vol_exp_real": m_real_B,
        "A2_vol_exp_shuffled": shuf_results_A2,
        "A4_blend_recompute": {"cagr": cagr, "sharpe": sharpe,
                                "sortino": sortino, "dd": dd, "n_days": len(common)},
    }, f, indent=2, default=str)
print(f"\n[saved] {out}")
