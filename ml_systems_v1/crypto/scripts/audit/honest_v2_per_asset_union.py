"""honest_v2_per_asset_union.py -- UNION architecture OOS validation.

UNION = regime-matched winner (priority 1) OR cousin set (priority 2) OR
        bucket fallback (priority 3 for unprofiled assets)

CONSTRAINTS (per user 2026-05-20):
  - K_MAX = 8 (max simultaneous positions)
  - per_asset_cap = 10% (max bet for any single asset)
  - total_deploy_cap = 60% (sum of all positions <= 60%, ALWAYS 40% cash buffer)
  - Maximize opportunities (relaxed inclusion, no over-conservatism)
  - Stop loss + let winners run (exit policy: tight_trail_5_3_14d)

ALLOCATION TIERS (per user intuition):
  Tier 1: own_bull fires       — multiplier 1.20
  Tier 2: own_chop fires       — multiplier 1.00
  Tier 3: own_bear fires       — multiplier 0.70
  Tier 4: own_crash fires      — multiplier 0.40

BUCKET VOL multipliers:
  BLUE     0.6 (low alpha per dollar)
  STEADY   0.9
  VOLATILE 1.15
  DEGEN    1.30 (high tail capture)

CELL QUALITY: cell_sharpe / median_cell_sharpe clipped [0.7, 1.3]

Output: runs/audit/MA_EMA_PROFILE_2026_05_20/OOS_UNION_REPORT.md
"""
# [!] SPLIT DISCIPLINE NOTE (2026-05-24 INST-C cleanup):
# This script uses the legacy convention where "OOS" labels the post-TRAIN window
# (= canonical OOS + UNSEEN combined). Per src/split_config.py the canonical OOS
# ends 2025-12-31 and UNSEEN starts 2026-01-01. The dates hardcoded below are
# intentionally preserved for reproducibility of prior outputs. New scripts must
# import from split_config -- see docs/SPLIT_DISCIPLINE.md.
from __future__ import annotations

import sys
import json
import glob
import yaml
from datetime import date as _date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / "data" / "processed" / "per_asset_ma_ema_profile.parquet"
WINNERS_PATH = ROOT / "data" / "processed" / "per_asset_regime_winners.parquet"
FALLBACK_PATH = ROOT / "data" / "processed" / "bucket_dna_fallback.parquet"
OWN_REGIME_PATH = ROOT / "data" / "processed" / "asset_own_regime_panel.parquet"
CHIMERA_1D = ROOT / "data" / "processed" / "chimera" / "1d"
U50_YAML = ROOT / "config" / "universes" / "u50.yaml"
U100_YAML = ROOT / "config" / "universes" / "u100.yaml"
OUT_DIR = ROOT / "runs" / "audit" / "MA_EMA_PROFILE_2026_05_20"

OOS_START = _date(2024, 5, 16)
OOS_END   = _date(2025, 3, 15)

# User-mandated constraints (2026-05-20)
K_MAX = 8
PER_ASSET_CAP = 0.10           # 10% per asset
TOTAL_DEPLOY_CAP = 0.60        # 60% portfolio max, 40% cash always
COST_RT = 0.0030

# Exit (best from prior OOS sweep)
EXIT_HOLD_MAX = 14
EXIT_STOP = -0.04
EXIT_TRAIL_ARM = 0.05
EXIT_TRAIL_DROP = 0.03

# Allocation multipliers (user-specified bull-tier + vol-profile bias)
REGIME_TIER_MULT = {"bull": 1.20, "chop": 1.00, "bear": 0.70, "crash": 0.40}
BUCKET_VOL_MULT = {"BLUE": 0.6, "STEADY": 0.9, "VOLATILE": 1.15, "DEGEN": 1.30}

# Regime qualification for cousin-set cells
REGIME_QUALIFIES = {
    "ALL_WEATHER":      {"bull", "chop", "bear", "crash"},
    "BLOCK_OWN_CRASH":  {"bull", "chop", "bear"},
    "BLOCK_OWN_BEAR":   {"bull", "chop"},
    "BULL_AND_CHOP":    {"bull", "chop"},
    "BULL_ONLY":        {"bull"},
    "REGIME_DEPENDENT": {"bull", "chop"},
    "INSUFFICIENT_DATA": set(),
}


def load_universe():
    """u100 yaml -> asset -> bucket mapping for ALL assets (profiled + unprofiled)."""
    with open(U50_YAML) as f:
        u50 = yaml.safe_load(f)
    with open(U100_YAML) as f:
        u100 = yaml.safe_load(f)
    asset_to_bucket = {}
    for a in u50["assets"]:
        sym = a["symbol"].replace("USDT", "")
        asset_to_bucket[sym] = a.get("dna", "VOLATILE")
    for a in u100.get("extra_assets", []):
        if a.get("status") != "ready": continue
        sym = a["symbol"].replace("USDT", "")
        asset_to_bucket[sym] = a.get("dna", "VOLATILE")
    return asset_to_bucket


def walk_forward_exit(entry_price, fwd_closes):
    peak = entry_price; armed = False
    for d, p in enumerate(fwd_closes, start=1):
        if p is None or not np.isfinite(p): continue
        ret = p / entry_price - 1
        if p > peak: peak = p
        if not armed and ret >= EXIT_TRAIL_ARM: armed = True
        if ret <= EXIT_STOP: return EXIT_STOP - COST_RT, d, "stop"
        if armed and p <= peak * (1 - EXIT_TRAIL_DROP):
            return p / entry_price - 1 - COST_RT, d, "trail"
        if d >= EXIT_HOLD_MAX: return ret - COST_RT, d, "max_hold"
    last = next((p for p in reversed(fwd_closes) if p is not None and np.isfinite(p)), None)
    if last is None: return -COST_RT, 0, "no_data"
    return last / entry_price - 1 - COST_RT, len(fwd_closes), "expire"


def compute_ma_active(chim, ma_type, fast, slow):
    """Compute boolean array: MA fast > slow at each bar (long signal active)."""
    closes = chim["close"].values
    if ma_type == "SMA":
        ma_f = pd.Series(closes).rolling(fast).mean().values
        ma_s = pd.Series(closes).rolling(slow).mean().values
    else:
        ma_f = pd.Series(closes).ewm(span=fast, adjust=False).mean().values
        ma_s = pd.Series(closes).ewm(span=slow, adjust=False).mean().values
    return ma_f > ma_s, ma_f, ma_s


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("="*78)
    print(f"UNION ARCHITECTURE OOS ({OOS_START} → {OOS_END})")
    print("="*78)
    print(f"Constraints: K_MAX={K_MAX}, per_asset_cap={PER_ASSET_CAP*100:.0f}%, "
          f"total_deploy_cap={TOTAL_DEPLOY_CAP*100:.0f}% (cash buffer {(1-TOTAL_DEPLOY_CAP)*100:.0f}%)")
    print(f"Allocation: regime tier × bucket vol × cell quality")
    print(f"Exit: hold≤{EXIT_HOLD_MAX}d, stop={EXIT_STOP*100:.0f}%, trail +{EXIT_TRAIL_ARM*100:.0f}%/-{EXIT_TRAIL_DROP*100:.0f}%")

    print("\nLoading inputs...")
    profile = pd.read_parquet(PROFILE_PATH)
    winners = pd.read_parquet(WINNERS_PATH)
    fallback = pd.read_parquet(FALLBACK_PATH)
    own_regime = pl.read_parquet(OWN_REGIME_PATH).to_pandas()
    own_regime["date"] = pd.to_datetime(own_regime["date"]).dt.normalize()
    asset_to_bucket = load_universe()

    cousins = profile[profile["is_cousin_set_member"]].copy()
    print(f"  cousins: {len(cousins)} across {cousins['asset'].nunique()} assets")
    print(f"  regime winners: {len(winners)} across {winners['asset'].nunique()} assets")
    print(f"  bucket fallback: {len(fallback)} buckets")
    print(f"  universe yaml: {len(asset_to_bucket)} assets")

    profiled = set(cousins["asset"].unique()) | set(winners["asset"].unique())
    unprofiled = set(asset_to_bucket.keys()) - profiled
    print(f"  profiled assets: {len(profiled)}; unprofiled (will use fallback): {len(unprofiled)}")

    # Index profile by asset for fast lookup
    cousins_by_asset = {a: g.to_dict("records") for a, g in cousins.groupby("asset")}
    winners_by_asset = {a: g.to_dict("records") for a, g in winners.groupby("asset")}
    fallback_by_bucket = {r["bucket"]: r.to_dict() for _, r in fallback.iterrows()}

    print("\nLoading chimera 1d...")
    chimera = {}
    for f in glob.glob(str(CHIMERA_1D / "*_v51_chimera_1d_*.parquet")):
        sym = Path(f).name.split("_")[0].upper().replace("USDT", "")
        try:
            df = pl.read_parquet(f, columns=["timestamp", "close"]).to_pandas()
        except Exception: continue
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.normalize()
        df = df.sort_values("date").reset_index(drop=True)
        chimera[sym] = df
    print(f"  {len(chimera)} assets with chimera data")

    # Build own-regime lookup (date->asset->regime)
    own_lookup = {(r["asset"], r["date"]): r["asset_own_regime"]
                  for _, r in own_regime.iterrows()}

    # For each asset, precompute fire flags for ALL its candidate cells
    print("\nPrecomputing MA-active matrices...")
    asset_cell_active = {}  # asset -> list of (cell_meta, active_bool_array)
    for asset in (set(asset_to_bucket.keys()) & set(chimera.keys())):
        chim = chimera[asset]
        cells_for_asset = []
        # Priority 1: regime winners
        for w in winners_by_asset.get(asset, []):
            active, _, _ = compute_ma_active(chim, w["ma_type"], int(w["fast"]), int(w["slow"]))
            cells_for_asset.append({
                "source": "regime_winner", "regime_qualifies": {w["own_regime"]},
                "ma_type": w["ma_type"], "fast": int(w["fast"]), "slow": int(w["slow"]),
                "sharpe": w["sharpe_in_regime"],
                "tier_priority": 1,  # highest priority
                "active": active,
            })
        # Priority 2: cousin cells
        for c in cousins_by_asset.get(asset, []):
            tag = c["regime_tag"]
            qualifies = REGIME_QUALIFIES.get(tag, set())
            active, _, _ = compute_ma_active(chim, c["ma_type"], int(c["fast"]), int(c["slow"]))
            cells_for_asset.append({
                "source": "cousin", "regime_qualifies": qualifies,
                "ma_type": c["ma_type"], "fast": int(c["fast"]), "slow": int(c["slow"]),
                "sharpe": c["sharpe"], "regime_tag": tag,
                "tier_priority": 2,
                "active": active,
            })
        # Priority 3: bucket fallback (only for unprofiled assets)
        if not cells_for_asset:
            bucket = asset_to_bucket.get(asset, "VOLATILE")
            fb = fallback_by_bucket.get(bucket)
            if fb is not None:
                active, _, _ = compute_ma_active(chim, fb["ma_type"], int(fb["fast"]), int(fb["slow"]))
                cells_for_asset.append({
                    "source": "bucket_fallback", "regime_qualifies": {"bull", "chop"},
                    "ma_type": fb["ma_type"], "fast": int(fb["fast"]), "slow": int(fb["slow"]),
                    "sharpe": fb["mean_sharpe_proxy"], "regime_tag": "BULL_AND_CHOP",
                    "tier_priority": 3,
                    "active": active,
                })
        asset_cell_active[asset] = (chim, cells_for_asset)
    print(f"  {len(asset_cell_active)} assets with at least one cell")
    print(f"  sources: regime_winner={sum(1 for a,(c,cells) in asset_cell_active.items() if any(x['source']=='regime_winner' for x in cells))}, " +
          f"cousin={sum(1 for a,(c,cells) in asset_cell_active.items() if any(x['source']=='cousin' for x in cells))}, " +
          f"fallback-only={sum(1 for a,(c,cells) in asset_cell_active.items() if all(x['source']=='bucket_fallback' for x in cells))}")

    # Median cell sharpe across the universe (for cell-quality normalization)
    all_sharpes = []
    for a, (c, cells) in asset_cell_active.items():
        for x in cells:
            all_sharpes.append(x["sharpe"])
    median_sharpe = float(np.median([s for s in all_sharpes if s > 0])) if all_sharpes else 0.1
    print(f"  median cell-sharpe (positive): {median_sharpe:+.4f}")

    # Calendar dates
    cur = OOS_START; cal_dates = []
    while cur <= OOS_END:
        cal_dates.append(cur); cur += timedelta(days=1)

    print(f"\nSimulating {len(cal_dates)} days...")
    portfolio_value = 1.0
    available_cash = 1.0
    open_positions = []
    trade_log = []
    daily_records = []

    for sim_date in cal_dates:
        sim_dt = pd.Timestamp(sim_date)

        # ===== Close positions =====
        new_open = []
        for pos in open_positions:
            chim, _ = asset_cell_active.get(pos["asset"], (None, []))
            if chim is None:
                new_open.append(pos); continue
            fwd = chim[(chim["date"] > pos["entry_date"]) & (chim["date"] <= sim_dt)]
            fwd_closes = [float(p) if np.isfinite(p) else None for p in fwd["close"].values]
            if not fwd_closes:
                new_open.append(pos); continue
            rret, d_held, reason = walk_forward_exit(pos["entry_price"], fwd_closes)
            if reason in ("stop", "trail", "max_hold"):
                pnl = pos["bet_size"] * rret
                available_cash += pos["bet_size"] + pnl
                trade_log.append({
                    "asset": pos["asset"], "regime": pos.get("regime_at_entry", "?"),
                    "source": pos.get("source", "?"),
                    "entry_date": pos["entry_date"], "exit_date": sim_date,
                    "days_held": d_held, "bet_size": pos["bet_size"],
                    "realized_ret": rret, "exit_reason": reason,
                })
            else:
                new_open.append(pos)
        open_positions = new_open

        # ===== Identify today's candidates =====
        # For each asset:
        #   Determine asset's own regime today
        #   For each cell (regime_winner / cousin / fallback):
        #     If today's own_regime ∈ cell.regime_qualifies AND cell is active today: candidate
        #   Pick the highest-priority candidate per asset (regime_winner > cousin > fallback);
        #   tiebreak by cell sharpe

        open_assets = set(p["asset"] for p in open_positions)
        candidates = []
        for asset, (chim, cells) in asset_cell_active.items():
            if asset in open_assets: continue
            # Find this asset's idx in its chimera for today (pandas timestamps)
            date_mask = chim["date"] == sim_dt
            if not date_mask.any(): continue
            idx = int(np.where(date_mask)[0][0])
            own_r = own_lookup.get((asset, sim_dt))
            if own_r is None: continue

            # Walk cells in priority order
            best_cell = None
            for cell in sorted(cells, key=lambda c: c["tier_priority"]):
                if own_r not in cell["regime_qualifies"]: continue
                if not cell["active"][idx]: continue
                if best_cell is None or cell["tier_priority"] < best_cell["tier_priority"]:
                    best_cell = cell
                if best_cell is not None and best_cell["tier_priority"] == 1:
                    break  # priority 1 beats all
            if best_cell is None: continue

            bucket = asset_to_bucket.get(asset, "VOLATILE")
            # Composite deploy score
            tier_mult = REGIME_TIER_MULT.get(own_r, 0.5)
            vol_mult = BUCKET_VOL_MULT.get(bucket, 1.0)
            quality_mult = float(np.clip(best_cell["sharpe"] / max(median_sharpe, 0.05), 0.7, 1.3))
            deploy_score = best_cell["sharpe"] * tier_mult * vol_mult * quality_mult

            candidates.append({
                "asset": asset, "bucket": bucket, "regime": own_r,
                "source": best_cell["source"],
                "cell_sharpe": best_cell["sharpe"],
                "deploy_score": deploy_score,
                "tier_mult": tier_mult, "vol_mult": vol_mult, "quality_mult": quality_mult,
                "idx": idx,
                "chim": chim,
            })

        # ===== Rank + allocate =====
        if candidates:
            candidates.sort(key=lambda x: -x["deploy_score"])
            slots_remaining = K_MAX - len(open_positions)
            budget_remaining = portfolio_value * TOTAL_DEPLOY_CAP - sum(p["bet_size"] for p in open_positions)

            for cand in candidates:
                if slots_remaining <= 0: break
                if budget_remaining <= 0.001 * portfolio_value: break  # less than 0.1%

                # Position size = base × tier × vol × quality, capped at PER_ASSET_CAP × portfolio
                base = (TOTAL_DEPLOY_CAP / K_MAX) * portfolio_value  # 60%/8 = 7.5% per position avg
                target_size = base * cand["tier_mult"] * cand["vol_mult"] * cand["quality_mult"]
                hard_cap = portfolio_value * PER_ASSET_CAP
                actual_size = min(target_size, hard_cap, budget_remaining, available_cash)
                if actual_size < 0.005 * portfolio_value:  # less than 0.5% — skip
                    continue

                # Entry price = today's close
                today_row = cand["chim"][cand["chim"]["date"] == sim_dt]
                if today_row.empty: continue
                ep = float(today_row.iloc[0]["close"])
                if ep <= 0 or not np.isfinite(ep): continue

                available_cash -= actual_size
                budget_remaining -= actual_size
                slots_remaining -= 1
                open_positions.append({
                    "asset": cand["asset"], "entry_date": sim_dt,
                    "entry_price": ep, "bet_size": actual_size,
                    "regime_at_entry": cand["regime"],
                    "source": cand["source"],
                })

        # ===== MtM =====
        omtm = 0
        for pos in open_positions:
            chim, _ = asset_cell_active.get(pos["asset"], (None, []))
            if chim is None: omtm += pos["bet_size"]; continue
            av = chim[chim["date"] <= sim_dt]
            if not len(av): omtm += pos["bet_size"]; continue
            cp = float(av.iloc[-1]["close"])
            if not np.isfinite(cp): omtm += pos["bet_size"]; continue
            omtm += pos["bet_size"] * (cp / pos["entry_price"])
        portfolio_value = available_cash + omtm
        daily_records.append({
            "date": sim_date, "portfolio_value": portfolio_value,
            "n_open": len(open_positions),
            "deployed_pct": (portfolio_value - available_cash) / portfolio_value * 100,
        })

    daily_df = pd.DataFrame(daily_records)
    trades_df = pd.DataFrame(trade_log)

    # Metrics
    pv = daily_df["portfolio_value"].values
    total_pct = (pv[-1] / pv[0] - 1) * 100
    window_days = (OOS_END - OOS_START).days
    ann_pct = ((1 + total_pct/100) ** (365/max(window_days,1)) - 1) * 100
    drs = pv[1:] / pv[:-1] - 1
    sortino = (drs.mean() / drs[drs < 0].std() * np.sqrt(252)) if (drs < 0).sum() and drs[drs<0].std() > 0 else 0
    sharpe = (drs.mean() / drs.std() * np.sqrt(252)) if drs.std() > 0 else 0
    cum = pv / pv[0]; cm = np.maximum.accumulate(cum)
    max_dd = ((cum / cm - 1) * 100).min()
    calmar = ann_pct / abs(max_dd) if max_dd != 0 else 0
    mean_deployed = daily_df["deployed_pct"].mean()

    print(f"\n=== UNION OOS RESULTS ===")
    print(f"  Window: {OOS_START} → {OOS_END} ({window_days} days)")
    print(f"  Total return:   {total_pct:+.2f}%")
    print(f"  Annualized:     {ann_pct:+.2f}%")
    print(f"  Daily mean:     {drs.mean()*100:+.4f}%")
    print(f"  Sortino:        {sortino:+.3f}")
    print(f"  Sharpe:         {sharpe:+.3f}")
    print(f"  Max DD:         {max_dd:+.2f}%")
    print(f"  Calmar:         {calmar:+.3f}")
    print(f"  Trades:         {len(trades_df)}")
    print(f"  Mean deployed:  {mean_deployed:.1f}% (cash buffer {100-mean_deployed:.1f}%)")
    if len(trades_df):
        win_rate = (trades_df["realized_ret"] > 0).mean() * 100
        avg_win = trades_df.loc[trades_df["realized_ret"]>0, "realized_ret"].mean()*100 if (trades_df["realized_ret"]>0).any() else 0
        avg_loss = trades_df.loc[trades_df["realized_ret"]<0, "realized_ret"].mean()*100 if (trades_df["realized_ret"]<0).any() else 0
        max_win = trades_df["realized_ret"].max() * 100
        print(f"  Win rate:       {win_rate:.1f}%")
        print(f"  Avg W/L:        {avg_win:+.2f}% / {avg_loss:+.2f}%")
        print(f"  Max win:        {max_win:+.2f}%")
        print(f"  Per-source:")
        for src, g in trades_df.groupby("source"):
            sm = g["realized_ret"].sum() * 100
            print(f"    {src:<18} n={len(g):3d}  win={(g['realized_ret']>0).mean()*100:5.1f}%  sum_ret={sm:+.2f}%")
        print(f"  Per-regime:")
        for r, g in trades_df.groupby("regime"):
            sm = g["realized_ret"].sum() * 100
            print(f"    own_{r:<7} n={len(g):3d}  win={(g['realized_ret']>0).mean()*100:5.1f}%  sum_ret={sm:+.2f}%")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    daily_df.to_csv(OUT_DIR / "oos_union_daily.csv", index=False)
    if len(trades_df):
        trades_df.to_csv(OUT_DIR / "oos_union_trades.csv", index=False)

    lines = [
        f"# UNION Architecture OOS Validation (2026-05-20)\n",
        f"**Window**: {OOS_START} → {OOS_END} ({window_days} days, canonical OOS)",
        f"**Universe**: {len(asset_cell_active)} assets ({len(profiled)} profiled + {len(unprofiled)} fallback)",
        f"**Architecture**: regime-matched winner (priority 1) → cousin set (priority 2) → bucket fallback (priority 3)",
        f"**Constraints**: K≤{K_MAX}, per_asset≤{PER_ASSET_CAP*100:.0f}%, total_deploy≤{TOTAL_DEPLOY_CAP*100:.0f}% (cash buffer ≥{(1-TOTAL_DEPLOY_CAP)*100:.0f}%)",
        f"**Exit**: trail +{EXIT_TRAIL_ARM*100:.0f}%/-{EXIT_TRAIL_DROP*100:.0f}%, hold≤{EXIT_HOLD_MAX}d, stop {EXIT_STOP*100:.0f}%",
        "",
        "## Headline",
        f"- **Total return**: **{total_pct:+.2f}%**",
        f"- **Annualized**: **{ann_pct:+.2f}%**",
        f"- **Daily compound**: **{drs.mean()*100:+.4f}%/d**",
        f"- Sortino: {sortino:+.3f} ; Sharpe: {sharpe:+.3f} ; Calmar: {calmar:+.3f}",
        f"- Max DD: {max_dd:+.2f}%",
        f"- Trades: {len(trades_df)} ; Mean deployed: {mean_deployed:.1f}% (cash buffer {100-mean_deployed:.1f}%)",
        "",
        "## Comparison",
        "",
        "| Architecture | OOS NAV | Annualized | Daily | Sortino | Max DD | Constraint |",
        "|---|---:|---:|---:|---:|---:|---|",
        f"| **UNION + bucket fallback (this)** | **{total_pct:+.2f}%** | **{ann_pct:+.2f}%** | **{drs.mean()*100:+.4f}%/d** | **{sortino:+.3f}** | **{max_dd:+.2f}%** | **K=8 / 10% pa / 60% total** |",
        f"| per_asset_cousin_set (tight_trail) | +159.33% | +215.17% | +0.3518%/d | +4.176 | -19.12% | K=12 / equal-wt |",
        f"| per_asset_regime_routed | +89.10% | +115.43% | +0.2421%/d | +2.822 | -19.87% | K=12 / equal-wt |",
        f"| confluence_only (current shipped) | +67.21% | +85.76% | +0.236%/d | +1.563 | -31.91% | K=12 / 8.3% equal |",
        f"| random-K | +60.00% | +76.16% | +0.205%/d | +1.473 | -36.41% | K=12 / equal |",
        f"| best-K (perfect foresight) | +415.97% | +621.85% | +0.55%/d | +3.450 | -33.39% | K=12 / equal |",
        "",
    ]
    if len(trades_df):
        lines.append("## Trade attribution by source")
        lines.append("")
        lines.append("| source | trades | win % | sum_ret |")
        lines.append("|---|---:|---:|---:|")
        for src, g in trades_df.groupby("source"):
            lines.append(f"| {src} | {len(g)} | {(g['realized_ret']>0).mean()*100:.1f}% | {g['realized_ret'].sum()*100:+.2f}% |")
        lines.append("")
        lines.append("## Trade attribution by regime at entry")
        lines.append("")
        lines.append("| regime | trades | win % | sum_ret |")
        lines.append("|---|---:|---:|---:|")
        for r, g in trades_df.groupby("regime"):
            lines.append(f"| own_{r} | {len(g)} | {(g['realized_ret']>0).mean()*100:.1f}% | {g['realized_ret'].sum()*100:+.2f}% |")

    (OUT_DIR / "OOS_UNION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] wrote {OUT_DIR / 'OOS_UNION_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
