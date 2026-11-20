"""honest_v2_per_asset_best_cadence.py -- multi-cadence sleeve where each asset
deploys at its OWN best cadence.

Per `best_cadence_per_asset.csv`:
  - FLOKI: 4h
  - PEPE: 15m
  - SUI: 15m
  - ARB: 15m
  - OP: 1d
  - ETH: 1d
  - BTC: 1d
  - ... varies asset by asset

ARCHITECTURE:
  For each asset:
    1. Look up its winning_cadence from best_cadence_per_asset.csv
    2. Get its top-K cells from pair_by_asset_cadence at THAT cadence
    3. Load chimera at THAT cadence for the asset
    4. Sim with the asset's own cadence (multiple cadences in same portfolio)

  Constraints same: K=8, per-asset 10%, total 60%
  Exit: per-cadence tuned (using best-from-multi_cadence_explorer.py):
    1d:  trail +5%/-3%, hold 14 bars (14d) — same as production
    4h:  trail +10%/-5%, hold 60 bars (10d)
    1h:  trail +15%/-8%, hold 120 bars (5d)
    15m: trail +10%/-5%, hold 192 bars (2d)

  Confirmation gate: ≥2 qualifying cells per asset

Predict: PER-ASSET BEST CADENCE could be net positive if the per-asset cadence-
preferences are real (the per-event Sharpe shows FLOKI 4h Sh 0.46, PEPE 15m Sh
0.36). Whether that translates to portfolio level (which the prior multi-cadence
explorer suggested 1d dominates universally) is the empirical question.
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
PER_ASSET_CAD = ROOT / "runs" / "oracle_layer3" / "ma_ema_per_asset_train" / "pair_by_asset_cadence.parquet"
BEST_CAD = ROOT / "runs" / "oracle_layer3" / "ma_ema_per_asset_train" / "best_cadence_per_asset.csv"
OWN_REGIME = ROOT / "data" / "processed" / "asset_own_regime_panel.parquet"
CHIMERA_DIR = ROOT / "data" / "processed" / "chimera"
U50_YAML = ROOT / "config" / "universes" / "u50.yaml"
U100_YAML = ROOT / "config" / "universes" / "u100.yaml"
OUT_DIR = ROOT / "runs" / "audit" / "MA_EMA_PROFILE_2026_05_20"

OOS_START = _date(2024, 5, 16)
OOS_END = _date(2025, 3, 15)

K_MAX = 8
PER_ASSET_CAP = 0.10
TOTAL_DEPLOY_CAP = 0.60
COST_RT = 0.0030
TOP_K_CELLS_PER_ASSET = 5
MIN_N_SIGNALED = 20
MIN_HIT_RATE = 0.45
MIN_MEAN_PNL = 0.10

# Per-cadence best exit (from multi_cadence_explorer winner per cadence)
PER_CADENCE_EXIT = {
    "1d":  {"stop": -0.04, "arm": 0.05, "drop": 0.03, "hold_bars": 14},
    "4h":  {"stop": -0.04, "arm": 0.10, "drop": 0.05, "hold_bars": 60},
    "1h":  {"stop": -0.04, "arm": 0.15, "drop": 0.08, "hold_bars": 120},
    "15m": {"stop": -0.04, "arm": 0.10, "drop": 0.05, "hold_bars": 192},
}

REGIME_TIER_MULT = {"bull": 1.20, "chop": 1.00, "bear": 0.70, "crash": 0.40}
BUCKET_VOL_MULT = {"BLUE": 0.6, "STEADY": 0.9, "VOLATILE": 1.15, "DEGEN": 1.30}


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


def compute_ma(closes, ma_type, fast, slow):
    s = pd.Series(closes)
    if ma_type == "SMA":
        return s.rolling(fast).mean().values, s.rolling(slow).mean().values
    return s.ewm(span=fast, adjust=False).mean().values, s.ewm(span=slow, adjust=False).mean().values


def walk_forward_exit(entry_price, fwd_closes, hold_max, stop, arm, drop, cost):
    peak = entry_price; armed = False
    for d, p in enumerate(fwd_closes, start=1):
        if p is None or not np.isfinite(p): continue
        ret = p / entry_price - 1
        if p > peak: peak = p
        if not armed and ret >= arm: armed = True
        if ret <= stop: return stop - cost, d, "stop"
        if armed and p <= peak * (1 - drop):
            return p / entry_price - 1 - cost, d, "trail"
        if d >= hold_max: return ret - cost, d, "max_hold"
    last = next((p for p in reversed(fwd_closes) if p is not None and np.isfinite(p)), None)
    if last is None: return -cost, 0, "no_data"
    return last / entry_price - 1 - cost, len(fwd_closes), "expire"


def load_chimera_for_asset(asset: str, cadence: str):
    """Load chimera at the asset's best cadence."""
    chim_dir = CHIMERA_DIR / cadence
    sym_lower = asset.lower() + "usdt"
    if cadence == "1d":
        pat = f"{sym_lower}_v51_chimera_1d_*.parquet"
    else:
        pat = f"{sym_lower}_v51_chimera_{cadence}_*.parquet"
    files = sorted(glob.glob(str(chim_dir / pat)))
    if not files: return None
    try:
        df = pl.read_parquet(files[-1], columns=["timestamp", "close"]).to_pandas()
    except Exception: return None
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["date"] = df["timestamp_dt"].dt.normalize()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("="*78)
    print("PER-ASSET BEST-CADENCE MULTI-CADENCE OOS SIM")
    print("="*78)
    asset_to_bucket = load_universe()
    own_regime = pl.read_parquet(OWN_REGIME).to_pandas()
    own_regime["date"] = pd.to_datetime(own_regime["date"]).dt.normalize()
    own_lookup = {(r["asset"], r["date"]): r["asset_own_regime"] for _, r in own_regime.iterrows()}

    best_cad = pd.read_csv(BEST_CAD)
    print(f"Best-cadence assets: {len(best_cad)}")
    print(f"Cadence distribution: {best_cad['winning_cadence'].value_counts().to_dict()}")

    per_asset_cad = pl.read_parquet(PER_ASSET_CAD).to_pandas()

    # Build per-asset profile at its winning cadence + load its chimera
    print("\nBuilding per-asset best-cadence profile...")
    asset_data = {}  # asset -> {cadence, chim, cells (list), exit_cfg}
    n_skipped = 0
    for _, row in best_cad.iterrows():
        asset = row["asset"]; cadence = row["winning_cadence"]
        if asset not in asset_to_bucket:
            n_skipped += 1; continue
        if cadence not in PER_CADENCE_EXIT:
            n_skipped += 1; continue
        # Cells: top-K from pair_by_asset_cadence at (asset, cadence)
        sub = per_asset_cad[(per_asset_cad["asset"] == asset) &
                              (per_asset_cad["cadence"] == cadence)]
        qual = sub[(sub["n_signaled"] >= MIN_N_SIGNALED) &
                    (~sub["degenerate_signal"]) &
                    (~sub["signal_quasi_constant"]) &
                    (sub["hit_rate"] >= MIN_HIT_RATE) &
                    (sub["mean_pnl_pct"] >= MIN_MEAN_PNL)]
        if qual.empty:
            n_skipped += 1; continue
        # pair_by_asset_cadence columns: ma_type, fast, slow, sharpe_proxy
        top_cells = qual.nlargest(TOP_K_CELLS_PER_ASSET, "sharpe_proxy").to_dict("records")
        # Normalize column names so downstream code uses consistent keys
        for tc in top_cells:
            tc["best_ma_type"] = tc.get("ma_type")
            tc["best_fast"] = tc.get("fast")
            tc["best_slow"] = tc.get("slow")
            tc["best_sharpe_proxy"] = tc.get("sharpe_proxy")

        chim = load_chimera_for_asset(asset, cadence)
        if chim is None:
            n_skipped += 1; continue

        asset_data[asset] = {
            "cadence": cadence,
            "bucket": asset_to_bucket[asset],
            "chim": chim,
            "cells": top_cells,
            "exit_cfg": PER_CADENCE_EXIT[cadence],
        }
    print(f"  prepared: {len(asset_data)} assets ({n_skipped} skipped)")
    print(f"  cadence distribution in prepared set:")
    cad_count = {}
    for a, d in asset_data.items():
        cad_count[d["cadence"]] = cad_count.get(d["cadence"], 0) + 1
    for c, n in sorted(cad_count.items(), key=lambda x: -x[1]):
        print(f"    {c}: {n}")

    # Precompute MA active arrays per asset per cell
    print("\nComputing MA active matrices per asset...")
    for asset, d in asset_data.items():
        closes = d["chim"]["close"].values
        cells_compiled = []
        for c in d["cells"]:
            try:
                ma_f, ma_s = compute_ma(closes, c["best_ma_type"], int(c["best_fast"]), int(c["best_slow"]))
                cells_compiled.append({
                    "ma_type": c["best_ma_type"], "fast": int(c["best_fast"]), "slow": int(c["best_slow"]),
                    "sharpe": float(c["best_sharpe_proxy"]),
                    "active": ma_f > ma_s,
                })
            except Exception:
                pass
        d["cells_compiled"] = cells_compiled

    # Build per-asset (date → last-bar-idx) lookup (pd.Timestamp keys, FIX from prior bugs)
    for asset, d in asset_data.items():
        d_to_i = {}
        for i, dt_val in enumerate(d["chim"]["date"].values):
            d_to_i[pd.Timestamp(dt_val)] = i
        d["date_to_idx"] = d_to_i

    # Median sharpe across all cells
    all_sharpes = [c["sharpe"] for d in asset_data.values() for c in d["cells_compiled"]]
    median_sharpe = float(np.median([s for s in all_sharpes if s > 0])) if all_sharpes else 0.1

    cur = OOS_START; oos_dates = []
    while cur <= OOS_END:
        oos_dates.append(cur); cur += timedelta(days=1)

    print(f"\nSimulating {len(oos_dates)} OOS days...")
    portfolio = 1.0
    cash = 1.0
    open_pos = []
    trades = []
    daily = []

    for sim_d in oos_dates:
        sim_dt = pd.Timestamp(sim_d)

        # Close positions (each asset uses its own cadence's exit)
        new_open = []
        for pos in open_pos:
            d = asset_data.get(pos["asset"])
            if d is None: new_open.append(pos); continue
            today_idx = d["date_to_idx"].get(sim_dt)
            if today_idx is None: new_open.append(pos); continue
            ei = pos["entry_idx"]
            if today_idx <= ei: new_open.append(pos); continue
            fwd = d["chim"]["close"].values[ei + 1: today_idx + 1].tolist()
            fwd = [float(p) if np.isfinite(p) else None for p in fwd]
            ex = d["exit_cfg"]
            rret, dh, reason = walk_forward_exit(
                pos["entry_price"], fwd, ex["hold_bars"], ex["stop"], ex["arm"], ex["drop"], COST_RT,
            )
            if reason in ("stop", "trail", "max_hold"):
                pnl = pos["bet_size"] * rret
                cash += pos["bet_size"] + pnl
                trades.append({
                    "asset": pos["asset"], "cadence": d["cadence"],
                    "bars_held": dh, "bet_size": pos["bet_size"],
                    "realized_ret": rret, "exit_reason": reason,
                })
            else:
                new_open.append(pos)
        open_pos = new_open

        # Candidates (confirmation gate: ≥2 active cells)
        open_assets = set(p["asset"] for p in open_pos)
        candidates = []
        for asset, d in asset_data.items():
            if asset in open_assets: continue
            today_idx = d["date_to_idx"].get(sim_dt)
            if today_idx is None: continue
            own_r = own_lookup.get((asset, sim_dt))
            if own_r is None: continue
            active = [c for c in d["cells_compiled"] if c["active"][today_idx]]
            if len(active) < 2: continue  # confirmation gate
            best = max(active, key=lambda c: c["sharpe"])
            tier = REGIME_TIER_MULT.get(own_r, 0.5)
            vol = BUCKET_VOL_MULT.get(d["bucket"], 1.0)
            q = float(np.clip(best["sharpe"] / max(median_sharpe, 0.05), 0.7, 1.3))
            candidates.append({
                "asset": asset, "cadence": d["cadence"], "score": best["sharpe"] * tier * vol * q,
                "tier": tier, "vol": vol, "q": q,
                "chim": d["chim"], "today_idx": today_idx,
            })
        candidates.sort(key=lambda c: -c["score"])

        # Enter
        slots = K_MAX - len(open_pos)
        budget = portfolio * TOTAL_DEPLOY_CAP - sum(p["bet_size"] for p in open_pos)
        for c in candidates:
            if slots <= 0 or budget <= 0.001 * portfolio: break
            base = (TOTAL_DEPLOY_CAP / K_MAX) * portfolio
            target = base * c["tier"] * c["vol"] * c["q"]
            actual = min(target, portfolio * PER_ASSET_CAP, budget, cash)
            if actual < 0.005 * portfolio: continue
            ep = float(c["chim"].iloc[c["today_idx"]]["close"])
            if ep <= 0 or not np.isfinite(ep): continue
            cash -= actual; budget -= actual; slots -= 1
            open_pos.append({
                "asset": c["asset"], "entry_idx": c["today_idx"],
                "entry_price": ep, "bet_size": actual,
            })

        # MtM
        omtm = 0
        for pos in open_pos:
            d = asset_data.get(pos["asset"])
            if d is None: omtm += pos["bet_size"]; continue
            today_idx = d["date_to_idx"].get(sim_dt)
            if today_idx is None: omtm += pos["bet_size"]; continue
            cp = float(d["chim"].iloc[today_idx]["close"])
            if not np.isfinite(cp): omtm += pos["bet_size"]; continue
            omtm += pos["bet_size"] * (cp / pos["entry_price"])
        portfolio = cash + omtm
        daily.append({"date": sim_d, "pv": portfolio, "n_open": len(open_pos)})

    daily_df = pd.DataFrame(daily)
    trades_df = pd.DataFrame(trades)

    pv = daily_df["pv"].values
    total = (pv[-1] / pv[0] - 1) * 100
    win_days = (OOS_END - OOS_START).days
    ann = ((1 + total/100) ** (365/max(win_days,1)) - 1) * 100
    drs = pv[1:] / pv[:-1] - 1
    sortino = (drs.mean() / drs[drs < 0].std() * np.sqrt(252)) if (drs < 0).sum() and drs[drs<0].std() > 0 else 0
    cum = pv / pv[0]; cm = np.maximum.accumulate(cum)
    max_dd = ((cum / cm - 1) * 100).min()

    print(f"\n=== PER-ASSET BEST-CADENCE OOS RESULTS ===")
    print(f"  Total: {total:+.2f}%  Annualized: {ann:+.2f}%  Daily: {drs.mean()*100:+.4f}%")
    print(f"  Sortino: {sortino:+.3f}  Max DD: {max_dd:+.2f}%")
    print(f"  Trades: {len(trades_df)}")
    if len(trades_df):
        wr = (trades_df["realized_ret"] > 0).mean() * 100
        aw = trades_df.loc[trades_df["realized_ret"]>0, "realized_ret"].mean()*100 if (trades_df["realized_ret"]>0).any() else 0
        al = trades_df.loc[trades_df["realized_ret"]<0, "realized_ret"].mean()*100 if (trades_df["realized_ret"]<0).any() else 0
        mw = trades_df["realized_ret"].max() * 100
        print(f"  Win rate: {wr:.1f}%  Avg W/L: {aw:+.2f}% / {al:+.2f}%  Max win: {mw:+.2f}%")
        print(f"  Per-cadence trades:")
        for cad, g in trades_df.groupby("cadence"):
            cad_total = (g["realized_ret"] * g["bet_size"]).sum() * 100
            print(f"    {cad}: {len(g)} trades, sum NAV contrib: {cad_total:+.2f}%")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "mode": "per_asset_best_cadence",
        "total_pct": float(total), "ann_pct": float(ann),
        "daily_mean_pct": float(drs.mean() * 100) if len(drs) else 0,
        "sortino": float(sortino), "max_dd_pct": float(max_dd),
        "n_trades": int(len(trades_df)),
        "win_rate_pct": float((trades_df["realized_ret"] > 0).mean() * 100) if len(trades_df) else 0,
        "n_assets_deployed": len(asset_data),
        "cadence_distribution": cad_count,
    }
    json.dump(out, (OUT_DIR / "per_asset_best_cadence_metrics.json").open("w"), indent=2, default=str)

    if len(trades_df):
        trades_df.to_csv(OUT_DIR / "per_asset_best_cadence_trades.csv", index=False)

    # Markdown
    lines = [
        f"# Per-Asset Best-Cadence Multi-Cadence Sleeve OOS (2026-05-20)\n",
        f"**Window**: {OOS_START} → {OOS_END}",
        f"**Architecture**: each asset deploys at its OWN winning cadence (from best_cadence_per_asset.csv)",
        f"**Universe**: {len(asset_data)} assets with valid cells; cadence dist: {cad_count}",
        f"**Constraints**: K=8, per-asset 10%, total 60%, confirmation gate (≥2 cells)",
        f"**Exits**: per-cadence tuned (1d: 5/3/14d; 4h: 10/5/60b; 1h: 15/8/120b; 15m: 10/5/192b)",
        "",
        "## Headline",
        f"- Total: **{total:+.2f}%**",
        f"- Annualized: **{ann:+.2f}%**",
        f"- Daily: **{drs.mean()*100:+.4f}%/d**",
        f"- Sortino: {sortino:+.3f}",
        f"- Max DD: {max_dd:+.2f}%",
        f"- Trades: {len(trades_df)}",
        "",
        "## Comparison",
        "",
        "| Architecture | OOS NAV | Sortino | DD | Trades | Notes |",
        "|---|---:|---:|---:|---:|---|",
        f"| **per_asset_best_cadence (THIS)** | **{total:+.2f}%** | **{sortino:+.3f}** | **{max_dd:+.2f}%** | **{len(trades_df)}** | each asset at its own cadence |",
        f"| per_asset 1d + confirmation (DEPLOYED) | +91.40% | +3.65 | -13.25% | 385 | universal 1d |",
        f"| per_asset 1d no-gate | +72.94% | +3.04 | -18.24% | 497 | universal 1d |",
        f"| multi_cadence_explorer 1d best | +63.54% | +3.08 | -20.70% | 496 | no regime-tag |",
        f"| multi_cadence_explorer 4h best | +20.24% | +1.47 | -27.51% | 672 | no regime-tag |",
    ]
    if len(trades_df):
        lines += ["", "## Per-cadence trade attribution", "",
                  "| Cadence | Trades | NAV contribution (bet × ret) % |",
                  "|---|---:|---:|"]
        for cad, g in trades_df.groupby("cadence"):
            cad_total = (g["realized_ret"] * g["bet_size"]).sum() * 100
            lines.append(f"| {cad} | {len(g)} | {cad_total:+.2f}% |")

    (OUT_DIR / "PER_ASSET_BEST_CADENCE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] wrote {OUT_DIR / 'PER_ASSET_BEST_CADENCE_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
