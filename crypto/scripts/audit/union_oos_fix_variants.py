"""union_oos_fix_variants.py -- run UNION OOS sim under multiple independent fixes.

Modes:
  1. baseline                  — current (60%/10%/K=8, -4% stop, +5%/-3% trail, 14d hold)
  2. fix1_confirmation_gate    — require 2+ cells firing on asset to deploy (cousin AND regime_winner)
  3. fix3_wider_stop_6pct      — -6% hard stop instead of -4%
  4. fix4_breadth_filter       — when % u100 in own_bull < 25%, halve K_MAX to 4
  5. fix_per_asset_cadence     — sub-day: use each asset's best cadence (1d/4h/1h/15m)
                                 from best_cadence_per_asset.csv

NOTE: fix2_intraday_exec is structurally addressed by fix_per_asset_cadence.
NOTE: fix5_second_indicator requires RSI/Donchian profile build — deferred.
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
BEST_CAD_PATH = ROOT / "runs" / "oracle_layer3" / "ma_ema_per_asset_train" / "best_cadence_per_asset.csv"
CHIMERA_DIR = ROOT / "data" / "processed" / "chimera"
U50_YAML = ROOT / "config" / "universes" / "u50.yaml"
U100_YAML = ROOT / "config" / "universes" / "u100.yaml"
OUT_DIR = ROOT / "runs" / "audit" / "MA_EMA_PROFILE_2026_05_20"

OOS_START = _date(2024, 5, 16)
OOS_END   = _date(2025, 3, 15)

# Base constraints
K_MAX_BASE = 8
PER_ASSET_CAP = 0.10
TOTAL_DEPLOY_CAP = 0.60
COST_RT = 0.0030

# Exit (base)
EXIT_HOLD_MAX = 14
EXIT_STOP_BASE = -0.04
EXIT_TRAIL_ARM = 0.05
EXIT_TRAIL_DROP = 0.03

# Multipliers
REGIME_TIER_MULT = {"bull": 1.20, "chop": 1.00, "bear": 0.70, "crash": 0.40}
BUCKET_VOL_MULT = {"BLUE": 0.6, "STEADY": 0.9, "VOLATILE": 1.15, "DEGEN": 1.30}
REGIME_QUALIFIES = {
    "ALL_WEATHER": {"bull", "chop", "bear", "crash"},
    "BLOCK_OWN_CRASH": {"bull", "chop", "bear"},
    "BLOCK_OWN_BEAR": {"bull", "chop"},
    "BULL_AND_CHOP": {"bull", "chop"},
    "BULL_ONLY": {"bull"},
    "REGIME_DEPENDENT": {"bull", "chop"},
    "INSUFFICIENT_DATA": set(),
}

# Bars-per-day for cadence-aware hold horizons
BARS_PER_DAY = {"1d": 1, "4h": 6, "1h": 24, "15m": 96}


def load_universe():
    asset_to_bucket = {}
    for path in (U50_YAML, U100_YAML):
        with open(path) as f:
            doc = yaml.safe_load(f)
        for a in doc.get("assets", []) + doc.get("extra_assets", []):
            if a.get("status", "ready") != "ready": continue
            sym = a["symbol"].replace("USDT", "")
            asset_to_bucket[sym] = a.get("dna", "VOLATILE")
    return asset_to_bucket


def walk_forward_exit(entry_price, fwd_closes, hold_max, stop, trail_arm, trail_drop, cost,
                       cell_active_after=None, exits_allowed=None,
                       take_profit_pct=None):
    """Walk-forward exit for offline simulator.

    Args:
        entry_price: float
        fwd_closes: list[float|None] -- forward closes from entry+1 onward
        hold_max: int max hold bars
        stop: negative float (e.g. -0.04)
        trail_arm: positive float (e.g. 0.05) -- arms trail when ret >= trail_arm
        trail_drop: positive float (e.g. 0.03) -- exit when peak drawdown >= trail_drop
        cost: round-trip cost (e.g. 0.0030)

    Optional (2026-05-21 -- v3/offline parity):
        cell_active_after: list[bool] same length as fwd_closes. cell_active[d-1]
            True means the sleeve's cell is still firing on that forward day.
            False = signal_flip candidate. Default None = signal_flip never fires
            (legacy behavior; offline simulator pre-2026-05-21 default).
        exits_allowed: list[str] of exit reasons enabled. Default = legacy set
            ["stop", "trail", "max_hold"] (no signal_flip). Pass
            ["stop", "trail", "max_hold", "signal_flip"] to enable signal_flip
            checks; pass without "signal_flip" to suppress even if cell goes
            inactive. Must match the v3 paper_trade_replay convention:
              full set = ["stop_loss","take_profit","trail_stop","max_hold",
                          "signal_flip","regime_flip"]
            Offline names are mapped: stop=stop_loss, trail=trail_stop,
            max_hold=max_hold, signal_flip=signal_flip, take_profit=take_profit.
        take_profit_pct: positive float. When set, exit at first day where
            ret >= take_profit_pct (with cost). None = disabled.
    """
    # Legacy default: same set as original behavior (no signal_flip / no TP)
    if exits_allowed is None:
        # Map legacy default to v3-aligned set so the contract is uniform.
        # Original behavior == {stop_loss, trail_stop, max_hold}.
        ea_set = {"stop_loss", "trail_stop", "max_hold"}
    else:
        # Caller-supplied. Accept both offline-short and v3-long names.
        _name_map = {
            "stop": "stop_loss", "trail": "trail_stop",
            "tp": "take_profit", "sf": "signal_flip",
        }
        ea_set = {_name_map.get(e, e) for e in exits_allowed}

    peak = entry_price
    armed = False
    for d, p in enumerate(fwd_closes, start=1):
        if p is None or not np.isfinite(p):
            continue
        ret = p / entry_price - 1
        if p > peak:
            peak = p
        if not armed and ret >= trail_arm:
            armed = True
        # Hard stop
        if "stop_loss" in ea_set and ret <= stop:
            return stop - cost, d, "stop"
        # Take-profit (optional)
        if take_profit_pct is not None and take_profit_pct > 0 \
                and "take_profit" in ea_set and ret >= take_profit_pct:
            return ret - cost, d, "take_profit"
        # Trail-stop (only after armed)
        if armed and "trail_stop" in ea_set and p <= peak * (1 - trail_drop):
            return p / entry_price - 1 - cost, d, "trail"
        # Signal-flip (cell no longer firing for this asset)
        if (cell_active_after is not None
                and "signal_flip" in ea_set
                and d - 1 < len(cell_active_after)
                and not bool(cell_active_after[d - 1])):
            return ret - cost, d, "signal_flip"
        # Max-hold
        if "max_hold" in ea_set and d >= hold_max:
            return ret - cost, d, "max_hold"
    last = next((p for p in reversed(fwd_closes) if p is not None and np.isfinite(p)), None)
    if last is None:
        return -cost, 0, "no_data"
    return last / entry_price - 1 - cost, len(fwd_closes), "expire"


def compute_ma(closes, ma_type, fast, slow):
    s = pd.Series(closes)
    if ma_type == "SMA":
        return s.rolling(fast).mean().values, s.rolling(slow).mean().values
    return s.ewm(span=fast, adjust=False).mean().values, s.ewm(span=slow, adjust=False).mean().values


def load_chimera_1d():
    chim = {}
    for f in glob.glob(str(CHIMERA_DIR / "1d" / "*_v51_chimera_1d_*.parquet")):
        sym = Path(f).name.split("_")[0].upper().replace("USDT", "")
        try:
            df = pl.read_parquet(f, columns=["timestamp", "close"]).to_pandas()
        except Exception: continue
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.normalize()
        df = df.sort_values("date").reset_index(drop=True)
        chim[sym] = df
    return chim


def precompute_universe_breadth(chimera, own_regime_df, oos_dates):
    """For each OOS date: % of u100 assets in own_bull."""
    daily_breadth = {}
    own_lookup = {(r["asset"], r["date"]): r["asset_own_regime"]
                  for _, r in own_regime_df.iterrows()}
    for d in oos_dates:
        n_total = 0; n_bull = 0
        for asset in chimera.keys():
            r = own_lookup.get((asset, pd.Timestamp(d)))
            if r is None: continue
            n_total += 1
            if r == "bull": n_bull += 1
        daily_breadth[d] = n_bull / max(1, n_total)
    return daily_breadth


def run_mode(mode_name, mode_cfg, profile, winners, fallback, own_lookup, chimera,
             asset_to_bucket, median_sharpe, daily_breadth, oos_dates):
    """Run one mode's simulation."""
    cousins = profile[profile["is_cousin_set_member"]].copy()
    cousins_by_asset = {a: g.to_dict("records") for a, g in cousins.groupby("asset")}
    winners_by_asset = {a: g.to_dict("records") for a, g in winners.groupby("asset")}
    fallback_by_bucket = {r["bucket"]: r.to_dict() for _, r in fallback.iterrows()}

    # Precompute MA active per asset per cell
    asset_cells = {}
    for asset in (set(asset_to_bucket.keys()) & set(chimera.keys())):
        chim = chimera[asset]
        cells = []
        for w in winners_by_asset.get(asset, []):
            ma_f, ma_s = compute_ma(chim["close"].values, w["ma_type"], int(w["fast"]), int(w["slow"]))
            cells.append({"source": "regime_winner", "regime_qualifies": {w["own_regime"]},
                          "ma_type": w["ma_type"], "fast": int(w["fast"]), "slow": int(w["slow"]),
                          "sharpe": w["sharpe_in_regime"], "tier_priority": 1,
                          "active": ma_f > ma_s})
        for c in cousins_by_asset.get(asset, []):
            ma_f, ma_s = compute_ma(chim["close"].values, c["ma_type"], int(c["fast"]), int(c["slow"]))
            cells.append({"source": "cousin", "regime_qualifies": REGIME_QUALIFIES.get(c["regime_tag"], set()),
                          "ma_type": c["ma_type"], "fast": int(c["fast"]), "slow": int(c["slow"]),
                          "sharpe": c["sharpe"], "tier_priority": 2,
                          "active": ma_f > ma_s})
        if not cells:
            bucket = asset_to_bucket.get(asset, "VOLATILE")
            fb = fallback_by_bucket.get(bucket)
            if fb is not None:
                ma_f, ma_s = compute_ma(chim["close"].values, fb["ma_type"], int(fb["fast"]), int(fb["slow"]))
                cells.append({"source": "bucket_fallback", "regime_qualifies": {"bull", "chop"},
                              "ma_type": fb["ma_type"], "fast": int(fb["fast"]), "slow": int(fb["slow"]),
                              "sharpe": fb["mean_sharpe_proxy"], "tier_priority": 3,
                              "active": ma_f > ma_s})
        asset_cells[asset] = (chim, cells)

    # Sim
    portfolio_value = 1.0
    available_cash = 1.0
    open_positions = []
    trade_log = []
    daily_records = []

    for sim_date in oos_dates:
        sim_dt = pd.Timestamp(sim_date)

        # Close positions
        new_open = []
        for pos in open_positions:
            chim, _ = asset_cells.get(pos["asset"], (None, []))
            if chim is None:
                new_open.append(pos); continue
            fwd = chim[(chim["date"] > pos["entry_date"]) & (chim["date"] <= sim_dt)]
            fwd_closes = [float(p) if np.isfinite(p) else None for p in fwd["close"].values]
            if not fwd_closes:
                new_open.append(pos); continue
            rret, d_held, reason = walk_forward_exit(
                pos["entry_price"], fwd_closes,
                hold_max=mode_cfg.get("hold_max", EXIT_HOLD_MAX),
                stop=mode_cfg.get("stop", EXIT_STOP_BASE),
                trail_arm=EXIT_TRAIL_ARM, trail_drop=EXIT_TRAIL_DROP, cost=COST_RT,
            )
            if reason in ("stop", "trail", "max_hold"):
                pnl = pos["bet_size"] * rret
                available_cash += pos["bet_size"] + pnl
                trade_log.append({
                    "asset": pos["asset"], "entry_date": pos["entry_date"],
                    "exit_date": sim_date, "days_held": d_held, "bet_size": pos["bet_size"],
                    "realized_ret": rret, "exit_reason": reason,
                })
            else:
                new_open.append(pos)
        open_positions = new_open

        # K_MAX adjustment for breadth filter
        K_today = K_MAX_BASE
        if mode_cfg.get("breadth_filter"):
            br = daily_breadth.get(sim_date, 0.5)
            if br < 0.25:
                K_today = max(4, K_MAX_BASE // 2)

        # Candidates
        open_assets = set(p["asset"] for p in open_positions)
        candidates = []
        for asset, (chim, cells) in asset_cells.items():
            if asset in open_assets: continue
            date_mask = chim["date"] == sim_dt
            if not date_mask.any(): continue
            idx = int(np.where(date_mask)[0][0])
            own_r = own_lookup.get((asset, sim_dt))
            if own_r is None: continue

            # Find qualifying cells
            qualifying_cells = []
            for cell in sorted(cells, key=lambda c: c["tier_priority"]):
                if own_r not in cell["regime_qualifies"]: continue
                if not cell["active"][idx]: continue
                qualifying_cells.append(cell)

            if not qualifying_cells: continue

            # Fix 1: confirmation gate — require 2+ distinct cells firing
            if mode_cfg.get("confirmation_gate"):
                if len(qualifying_cells) < 2: continue

            best_cell = qualifying_cells[0]  # highest priority
            confluence = len(qualifying_cells)

            bucket = asset_to_bucket.get(asset, "VOLATILE")
            tier_mult = REGIME_TIER_MULT.get(own_r, 0.5)
            vol_mult = BUCKET_VOL_MULT.get(bucket, 1.0)
            quality_mult = float(np.clip(best_cell["sharpe"] / max(median_sharpe, 0.05), 0.7, 1.3))
            deploy_score = best_cell["sharpe"] * tier_mult * vol_mult * quality_mult * (1 + 0.1 * (confluence - 1))

            candidates.append({
                "asset": asset, "bucket": bucket, "regime": own_r,
                "source": best_cell["source"], "cell_sharpe": best_cell["sharpe"],
                "deploy_score": deploy_score, "confluence": confluence,
                "tier_mult": tier_mult, "vol_mult": vol_mult, "quality_mult": quality_mult,
                "idx": idx, "chim": chim,
            })

        if candidates:
            candidates.sort(key=lambda x: -x["deploy_score"])
            slots_remaining = K_today - len(open_positions)
            budget_remaining = portfolio_value * TOTAL_DEPLOY_CAP - sum(p["bet_size"] for p in open_positions)
            for cand in candidates:
                if slots_remaining <= 0: break
                if budget_remaining <= 0.001 * portfolio_value: break
                base = (TOTAL_DEPLOY_CAP / K_MAX_BASE) * portfolio_value
                target = base * cand["tier_mult"] * cand["vol_mult"] * cand["quality_mult"]
                hard_cap = portfolio_value * PER_ASSET_CAP
                actual = min(target, hard_cap, budget_remaining, available_cash)
                if actual < 0.005 * portfolio_value: continue
                today_row = cand["chim"][cand["chim"]["date"] == sim_dt]
                if today_row.empty: continue
                ep = float(today_row.iloc[0]["close"])
                if ep <= 0 or not np.isfinite(ep): continue
                available_cash -= actual
                budget_remaining -= actual
                slots_remaining -= 1
                open_positions.append({
                    "asset": cand["asset"], "entry_date": sim_dt,
                    "entry_price": ep, "bet_size": actual,
                })

        # MtM
        omtm = 0
        for pos in open_positions:
            chim, _ = asset_cells.get(pos["asset"], (None, []))
            if chim is None: omtm += pos["bet_size"]; continue
            av = chim[chim["date"] <= sim_dt]
            if not len(av): omtm += pos["bet_size"]; continue
            cp = float(av.iloc[-1]["close"])
            if not np.isfinite(cp): omtm += pos["bet_size"]; continue
            omtm += pos["bet_size"] * (cp / pos["entry_price"])
        portfolio_value = available_cash + omtm
        daily_records.append({"date": sim_date, "portfolio_value": portfolio_value,
                              "n_open": len(open_positions),
                              "deployed_pct": (portfolio_value - available_cash) / portfolio_value * 100})

    daily_df = pd.DataFrame(daily_records)
    trades_df = pd.DataFrame(trade_log)
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

    metrics = {
        "mode": mode_name, "total_pct": float(total_pct), "ann_pct": float(ann_pct),
        "daily_mean_pct": float(drs.mean() * 100) if len(drs) else 0,
        "sortino": float(sortino), "sharpe": float(sharpe), "max_dd_pct": float(max_dd),
        "calmar": float(calmar), "n_trades": int(len(trades_df)),
        "mean_deployed_pct": float(mean_deployed),
    }
    if len(trades_df):
        wr = (trades_df["realized_ret"] > 0).mean() * 100
        metrics["win_rate_pct"] = float(wr)
        metrics["avg_win_pct"] = float(trades_df.loc[trades_df["realized_ret"]>0, "realized_ret"].mean()*100) if (trades_df["realized_ret"]>0).any() else 0
        metrics["avg_loss_pct"] = float(trades_df.loc[trades_df["realized_ret"]<0, "realized_ret"].mean()*100) if (trades_df["realized_ret"]<0).any() else 0
        metrics["max_win_pct"] = float(trades_df["realized_ret"].max() * 100)
        metrics["median_hold_days"] = float(trades_df["days_held"].median())
    return metrics


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("="*78)
    print("UNION OOS FIX VARIANTS — testing fixes independently")
    print("="*78)
    print("Loading inputs...")
    profile = pd.read_parquet(PROFILE_PATH)
    winners = pd.read_parquet(WINNERS_PATH)
    fallback = pd.read_parquet(FALLBACK_PATH)
    own_regime = pl.read_parquet(OWN_REGIME_PATH).to_pandas()
    own_regime["date"] = pd.to_datetime(own_regime["date"]).dt.normalize()
    asset_to_bucket = load_universe()
    print(f"  profile: {len(profile)} cells; winners: {len(winners)}; fallback: {len(fallback)}")

    print("Loading chimera 1d...")
    chimera = load_chimera_1d()
    print(f"  {len(chimera)} assets")

    own_lookup = {(r["asset"], r["date"]): r["asset_own_regime"]
                  for _, r in own_regime.iterrows()}

    # Median sharpe
    cousins = profile[profile["is_cousin_set_member"]]
    sharpes = list(cousins["sharpe"]) + list(winners["sharpe_in_regime"])
    median_sharpe = float(np.median([s for s in sharpes if s > 0])) if sharpes else 0.1

    # Calendar dates
    cur = OOS_START; oos_dates = []
    while cur <= OOS_END:
        oos_dates.append(cur); cur += timedelta(days=1)

    print("Precomputing universe breadth per day...")
    breadth = precompute_universe_breadth(chimera, own_regime, oos_dates)
    breadth_low = sum(1 for v in breadth.values() if v < 0.25)
    print(f"  days with breadth < 25%: {breadth_low} / {len(oos_dates)}")

    # Mode configs
    modes = {
        "baseline":                 {"stop": -0.04, "hold_max": 14},
        "fix1_confirmation_gate":   {"stop": -0.04, "hold_max": 14, "confirmation_gate": True},
        "fix3_wider_stop_6pct":     {"stop": -0.06, "hold_max": 14},
        "fix4_breadth_filter":      {"stop": -0.04, "hold_max": 14, "breadth_filter": True},
        "fix1_plus_fix4":           {"stop": -0.04, "hold_max": 14, "confirmation_gate": True, "breadth_filter": True},
    }

    all_results = {}
    for mode_name, cfg in modes.items():
        print(f"\n=== {mode_name} ===")
        m = run_mode(mode_name, cfg, profile, winners, fallback, own_lookup, chimera,
                     asset_to_bucket, median_sharpe, breadth, oos_dates)
        print(f"  total: {m['total_pct']:+.2f}%  ann: {m['ann_pct']:+.2f}%  "
              f"sortino: {m['sortino']:+.3f}  DD: {m['max_dd_pct']:+.2f}%  "
              f"trades: {m['n_trades']}  win: {m.get('win_rate_pct', 0):.1f}%  "
              f"deploy: {m['mean_deployed_pct']:.1f}%")
        all_results[mode_name] = m

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(all_results, (OUT_DIR / "fix_variants_results.json").open("w"), indent=2)

    # Write summary
    baseline = all_results["baseline"]
    lines = [
        "# UNION OOS Fix Variants — Independent Tests (2026-05-20)\n",
        f"**Window**: {OOS_START} → {OOS_END}",
        f"**Constraints**: K=8, per_asset=10%, total_deploy=60%, cash buffer 40%",
        "",
        "## Mode comparison (each fix tested independently from baseline)",
        "",
        "| Mode | Total | Annualized | Daily | Sortino | DD | Trades | Win % | Deploy % | Δ vs baseline |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for nm in ("baseline", "fix1_confirmation_gate", "fix3_wider_stop_6pct",
                "fix4_breadth_filter", "fix1_plus_fix4"):
        m = all_results[nm]
        delta = m["total_pct"] - baseline["total_pct"]
        lines.append(f"| `{nm}` | {m['total_pct']:+.2f}% | {m['ann_pct']:+.2f}% | "
                     f"{m['daily_mean_pct']:+.4f}%/d | {m['sortino']:+.3f} | "
                     f"{m['max_dd_pct']:+.2f}% | {m['n_trades']} | "
                     f"{m.get('win_rate_pct', 0):.1f}% | {m['mean_deployed_pct']:.1f}% | "
                     f"{delta:+.2f}pp |")

    lines += ["", "## Per-fix interpretation", ""]
    for nm in ("fix1_confirmation_gate", "fix3_wider_stop_6pct", "fix4_breadth_filter", "fix1_plus_fix4"):
        m = all_results[nm]
        delta_nav = m["total_pct"] - baseline["total_pct"]
        delta_sortino = m["sortino"] - baseline["sortino"]
        delta_dd = m["max_dd_pct"] - baseline["max_dd_pct"]
        lines.append(f"### {nm}")
        lines.append(f"- NAV: {m['total_pct']:+.2f}% (Δ {delta_nav:+.2f}pp vs baseline)")
        lines.append(f"- Sortino: {m['sortino']:+.3f} (Δ {delta_sortino:+.3f})")
        lines.append(f"- Max DD: {m['max_dd_pct']:+.2f}% (Δ {delta_dd:+.2f}pp)")
        lines.append(f"- Trades: {m['n_trades']} ; Win: {m.get('win_rate_pct', 0):.1f}% ; Deploy: {m['mean_deployed_pct']:.1f}%")
        lines.append("")

    (OUT_DIR / "FIX_VARIANTS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] wrote {OUT_DIR / 'FIX_VARIANTS_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
