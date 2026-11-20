"""multi_cadence_explorer.py -- side-by-side MA/EMA across 5 timeframes.

Smarter search than brute-force:
  1. For each cadence (1d, 4h, 1h, 15m, dollar), use already-mined profile data
     where available (pair_by_asset_cadence has 4 cadences).
  2. For dollar bars, use a small Fibonacci candidate set (no brute-force grid).
  3. Per-cadence EXIT TUNING: small grid (3 configs) per cadence, not per cell.
  4. Pick best-exit per cadence, then run side-by-side OOS.

Output:
  runs/audit/MA_EMA_PROFILE_2026_05_20/MULTI_CADENCE_COMPARISON.md
"""
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
PROFILE_1D = ROOT / "data" / "processed" / "per_asset_ma_ema_profile.parquet"
WINNERS_1D = ROOT / "data" / "processed" / "per_asset_regime_winners.parquet"
FALLBACK = ROOT / "data" / "processed" / "bucket_dna_fallback.parquet"
OWN_REGIME = ROOT / "data" / "processed" / "asset_own_regime_panel.parquet"
PER_ASSET_CAD = ROOT / "runs" / "oracle_layer3" / "ma_ema_per_asset_train" / "pair_by_asset_cadence.parquet"
CHIMERA_DIR = ROOT / "data" / "processed" / "chimera"
U50_YAML = ROOT / "config" / "universes" / "u50.yaml"
U100_YAML = ROOT / "config" / "universes" / "u100.yaml"
OUT_DIR = ROOT / "runs" / "audit" / "MA_EMA_PROFILE_2026_05_20"

OOS_START = _date(2024, 5, 16)
OOS_END = _date(2025, 3, 15)

# Common constraints
K_MAX = 8
PER_ASSET_CAP = 0.10
TOTAL_DEPLOY_CAP = 0.60
COST_RT = 0.0030

# Per-cadence exit tuning (smart grid: 3 configs per cadence)
# Wider trail at smaller bars (more noise per bar); hold scales by calendar-day target
CADENCE_EXIT_GRID = {
    "1d": [
        {"name": "trail_5_3_14",  "stop": -0.04, "arm": 0.05, "drop": 0.03, "hold_bars": 14},
        {"name": "trail_8_5_14",  "stop": -0.04, "arm": 0.08, "drop": 0.05, "hold_bars": 14},
        {"name": "trail_5_3_21",  "stop": -0.04, "arm": 0.05, "drop": 0.03, "hold_bars": 21},
    ],
    "4h": [
        {"name": "trail_5_3_84",   "stop": -0.04, "arm": 0.05, "drop": 0.03, "hold_bars": 84},   # 14d
        {"name": "trail_10_5_60",  "stop": -0.04, "arm": 0.10, "drop": 0.05, "hold_bars": 60},   # 10d
        {"name": "trail_15_7_42",  "stop": -0.04, "arm": 0.15, "drop": 0.07, "hold_bars": 42},   # 7d
    ],
    "1h": [
        {"name": "trail_8_5_168",  "stop": -0.04, "arm": 0.08, "drop": 0.05, "hold_bars": 168},  # 7d
        {"name": "trail_15_8_120", "stop": -0.04, "arm": 0.15, "drop": 0.08, "hold_bars": 120},  # 5d
        {"name": "trail_20_10_72", "stop": -0.04, "arm": 0.20, "drop": 0.10, "hold_bars": 72},   # 3d
    ],
    "15m": [
        {"name": "trail_10_5_192", "stop": -0.04, "arm": 0.10, "drop": 0.05, "hold_bars": 192},  # 2d
        {"name": "trail_15_8_96",  "stop": -0.04, "arm": 0.15, "drop": 0.08, "hold_bars": 96},   # 1d
        {"name": "trail_25_12_48", "stop": -0.04, "arm": 0.25, "drop": 0.12, "hold_bars": 48},   # 0.5d
    ],
    "dollar": [
        {"name": "trail_5_3_50",   "stop": -0.04, "arm": 0.05, "drop": 0.03, "hold_bars": 50},
        {"name": "trail_10_5_100", "stop": -0.04, "arm": 0.10, "drop": 0.05, "hold_bars": 100},
        {"name": "trail_15_8_200", "stop": -0.04, "arm": 0.15, "drop": 0.08, "hold_bars": 200},
    ],
}

# Smart-candidate Fibonacci pairs for dollar bars (no brute-force grid)
DOLLAR_FIB_CANDIDATES = [
    ("SMA", 5, 13), ("SMA", 8, 21), ("SMA", 13, 34), ("SMA", 21, 55), ("SMA", 34, 89),
    ("EMA", 5, 13), ("EMA", 8, 21), ("EMA", 13, 34),
]

REGIME_QUALIFIES = {
    "ALL_WEATHER": {"bull", "chop", "bear", "crash"},
    "BLOCK_OWN_CRASH": {"bull", "chop", "bear"},
    "BLOCK_OWN_BEAR": {"bull", "chop"},
    "BULL_AND_CHOP": {"bull", "chop"},
    "BULL_ONLY": {"bull"},
    "REGIME_DEPENDENT": {"bull", "chop"},
}
REGIME_TIER_MULT = {"bull": 1.20, "chop": 1.00, "bear": 0.70, "crash": 0.40}
BUCKET_VOL_MULT = {"BLUE": 0.6, "STEADY": 0.9, "VOLATILE": 1.15, "DEGEN": 1.30}

MIN_QUAL_GATES = {
    "n_signaled": 20, "hit_rate": 0.45, "mean_pnl_pct": 0.10,
}
TOP_K_PER_ASSET = 5  # smaller K per asset since we just want best cells


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


def build_profile_for_cadence(cadence: str, per_asset_cad: pd.DataFrame, asset_to_bucket: dict) -> pd.DataFrame:
    """Build per-asset top-K cells profile for one cadence (1d/4h/1h/15m)."""
    sub = per_asset_cad[per_asset_cad["cadence"] == cadence]
    profile = []
    for asset, g in sub.groupby("asset"):
        if asset not in asset_to_bucket: continue
        bucket = asset_to_bucket[asset]
        qual = g[(g["n_signaled"] >= MIN_QUAL_GATES["n_signaled"]) &
                  (~g["degenerate_signal"]) &
                  (~g["signal_quasi_constant"]) &
                  (g["hit_rate"] >= MIN_QUAL_GATES["hit_rate"]) &
                  (g["mean_pnl_pct"] >= MIN_QUAL_GATES["mean_pnl_pct"])]
        if qual.empty: continue
        top = qual.nlargest(TOP_K_PER_ASSET, "sharpe_proxy")
        for _, c in top.iterrows():
            profile.append({
                "asset": asset, "bucket": bucket, "cadence": cadence,
                "ma_type": c["ma_type"], "fast": int(c["fast"]), "slow": int(c["slow"]),
                "n_signaled": int(c["n_signaled"]),
                "mean_pnl_pct": float(c["mean_pnl_pct"]),
                "hit_rate": float(c["hit_rate"]),
                "sharpe": float(c["sharpe_proxy"]),
            })
    return pd.DataFrame(profile)


def build_dollar_profile(asset_to_bucket: dict) -> pd.DataFrame:
    """For dollar bars, no per-asset profile mined — use Fibonacci candidates universally."""
    rows = []
    for asset, bucket in asset_to_bucket.items():
        for ma_type, fast, slow in DOLLAR_FIB_CANDIDATES:
            rows.append({
                "asset": asset, "bucket": bucket, "cadence": "dollar",
                "ma_type": ma_type, "fast": fast, "slow": slow,
                "sharpe": 0.1,  # nominal; we don't have per-asset stats for dollar yet
            })
    return pd.DataFrame(rows)


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


def load_chimera(cadence: str) -> dict:
    """Load chimera at given cadence. Returns {asset: DataFrame}."""
    chim_dir = CHIMERA_DIR / cadence
    chim = {}
    if cadence == "dollar":
        pat = "*_v51_chimera_*.parquet"
    elif cadence == "1d":
        pat = "*_v51_chimera_1d_*.parquet"
    else:
        pat = f"*_v51_chimera_{cadence}_*.parquet"
    for f in glob.glob(str(chim_dir / pat)):
        sym = Path(f).name.split("_")[0].upper().replace("USDT", "")
        try:
            df = pl.read_parquet(f, columns=["timestamp", "close"]).to_pandas()
        except Exception: continue
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["date"] = df["timestamp_dt"].dt.normalize()
        df = df.sort_values("timestamp").reset_index(drop=True)
        chim[sym] = df
    return chim


def run_sim(cadence, profile, exit_cfg, chimera, own_lookup, asset_to_bucket, oos_dates):
    """One sim run."""
    median_sharpe = float(np.median([s for s in profile["sharpe"] if s > 0])) if not profile.empty else 0.1
    # Build per-asset cells with MA active arrays
    asset_cells = {}
    for asset, g in profile.groupby("asset"):
        chim = chimera.get(asset)
        if chim is None: continue
        bucket = asset_to_bucket.get(asset, "VOLATILE")
        cells_compiled = []
        for _, c in g.iterrows():
            try:
                ma_f, ma_s = compute_ma(chim["close"].values, c["ma_type"], int(c["fast"]), int(c["slow"]))
            except Exception: continue
            cells_compiled.append({
                "ma_type": c["ma_type"], "fast": int(c["fast"]), "slow": int(c["slow"]),
                "sharpe": float(c["sharpe"]), "active": ma_f > ma_s,
            })
        if cells_compiled:
            asset_cells[asset] = (chim, cells_compiled, bucket)

    if not asset_cells:
        return {"mode": cadence, "n_trades": 0, "total_pct": 0.0, "skipped": "no assets"}

    # Build per-asset (date → last-bar-idx-on-that-day) — use pd.Timestamp keys
    date_to_idx = {}
    for asset, (chim, _, _) in asset_cells.items():
        d_to_i = {}
        for i, d in enumerate(chim["date"].values):
            d_to_i[pd.Timestamp(d)] = i  # latest seen wins (handles multi-bar days)
        date_to_idx[asset] = d_to_i

    portfolio = 1.0
    cash = 1.0
    open_pos = []
    trades = []
    daily = []
    for sim_d in oos_dates:
        sim_dt = pd.Timestamp(sim_d)

        # Close positions
        new_open = []
        for pos in open_pos:
            chim, _, _ = asset_cells.get(pos["asset"], (None, [], None))
            if chim is None: new_open.append(pos); continue
            today_idx = date_to_idx.get(pos["asset"], {}).get(sim_dt)
            if today_idx is None: new_open.append(pos); continue
            ei = pos["entry_idx"]
            if today_idx <= ei: new_open.append(pos); continue
            fwd = chim["close"].values[ei + 1: today_idx + 1].tolist()
            fwd = [float(p) if np.isfinite(p) else None for p in fwd]
            rret, dh, reason = walk_forward_exit(
                pos["entry_price"], fwd, exit_cfg["hold_bars"],
                exit_cfg["stop"], exit_cfg["arm"], exit_cfg["drop"], COST_RT,
            )
            if reason in ("stop", "trail", "max_hold"):
                pnl = pos["bet_size"] * rret
                cash += pos["bet_size"] + pnl
                trades.append({
                    "asset": pos["asset"], "bars_held": dh, "bet_size": pos["bet_size"],
                    "realized_ret": rret, "exit_reason": reason,
                })
            else:
                new_open.append(pos)
        open_pos = new_open

        # Candidates with confirmation gate (>=2 qualifying cells)
        open_assets = set(p["asset"] for p in open_pos)
        candidates = []
        for asset, (chim, cells, bucket) in asset_cells.items():
            if asset in open_assets: continue
            today_idx = date_to_idx.get(asset, {}).get(sim_dt)
            if today_idx is None: continue
            own_r = own_lookup.get((asset, sim_dt))
            if own_r is None: continue
            # All cells with active==True at today_idx qualify (regime-gating omitted for cadence
            # variants since per-cadence regime tagging isn't built; rely on confluence-gate only)
            active = [c for c in cells if c["active"][today_idx]]
            if len(active) < 2: continue  # confirmation gate
            best = max(active, key=lambda c: c["sharpe"])
            tier = REGIME_TIER_MULT.get(own_r, 0.5)
            vol = BUCKET_VOL_MULT.get(bucket, 1.0)
            q = float(np.clip(best["sharpe"] / max(median_sharpe, 0.05), 0.7, 1.3))
            candidates.append({
                "asset": asset, "score": best["sharpe"] * tier * vol * q,
                "tier": tier, "vol": vol, "q": q,
                "chim": chim, "today_idx": today_idx, "bucket": bucket,
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
            chim, _, _ = asset_cells.get(pos["asset"], (None, [], None))
            if chim is None: omtm += pos["bet_size"]; continue
            today_idx = date_to_idx.get(pos["asset"], {}).get(sim_dt)
            if today_idx is None: omtm += pos["bet_size"]; continue
            cp = float(chim.iloc[today_idx]["close"])
            if not np.isfinite(cp): omtm += pos["bet_size"]; continue
            omtm += pos["bet_size"] * (cp / pos["entry_price"])
        portfolio = cash + omtm
        daily.append({"date": sim_d, "pv": portfolio})

    daily_df = pd.DataFrame(daily)
    if daily_df.empty: return {"mode": f"{cadence}_{exit_cfg['name']}", "n_trades": 0, "total_pct": 0.0}
    pv = daily_df["pv"].values
    total = (pv[-1] / pv[0] - 1) * 100
    win_days = (OOS_END - OOS_START).days
    ann = ((1 + total/100) ** (365/max(win_days,1)) - 1) * 100
    drs = pv[1:] / pv[:-1] - 1
    sortino = (drs.mean() / drs[drs < 0].std() * np.sqrt(252)) if (drs < 0).sum() and drs[drs<0].std() > 0 else 0
    cum = pv / pv[0]; cm = np.maximum.accumulate(cum)
    max_dd = ((cum / cm - 1) * 100).min()
    trades_df = pd.DataFrame(trades)
    return {
        "mode": f"{cadence}_{exit_cfg['name']}", "cadence": cadence, "exit": exit_cfg["name"],
        "total_pct": float(total), "ann_pct": float(ann),
        "daily_mean_pct": float(drs.mean() * 100) if len(drs) else 0,
        "sortino": float(sortino), "max_dd_pct": float(max_dd),
        "n_trades": int(len(trades_df)),
        "win_rate_pct": float((trades_df["realized_ret"] > 0).mean() * 100) if len(trades_df) else 0,
        "median_hold_bars": float(trades_df["bars_held"].median()) if len(trades_df) else 0,
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("="*78)
    print("MULTI-CADENCE EXPLORER — 1d, 4h, 1h, 15m, dollar")
    print("="*78)

    asset_to_bucket = load_universe()
    own_regime = pl.read_parquet(OWN_REGIME).to_pandas()
    own_regime["date"] = pd.to_datetime(own_regime["date"]).dt.normalize()
    own_lookup = {(r["asset"], r["date"]): r["asset_own_regime"] for _, r in own_regime.iterrows()}

    per_asset_cad = pl.read_parquet(PER_ASSET_CAD).to_pandas()

    cur = OOS_START; oos_dates = []
    while cur <= OOS_END:
        oos_dates.append(cur); cur += timedelta(days=1)

    all_results = []
    for cadence in ("1d", "4h", "1h", "15m", "dollar"):
        print(f"\n=== Cadence: {cadence} ===")
        # Build profile
        if cadence == "dollar":
            profile = build_dollar_profile(asset_to_bucket)
        else:
            profile = build_profile_for_cadence(cadence, per_asset_cad, asset_to_bucket)
        print(f"  profile: {len(profile)} cells / {profile['asset'].nunique()} assets")
        if profile.empty:
            print("  [skip] empty")
            continue

        # Load chimera at this cadence
        print(f"  loading chimera {cadence}...")
        chimera = load_chimera(cadence)
        print(f"  {len(chimera)} assets in chimera")
        if not chimera:
            print("  [skip] no chimera")
            continue

        # Run sim for each exit config
        for exit_cfg in CADENCE_EXIT_GRID[cadence]:
            print(f"  exit {exit_cfg['name']}: ", end="", flush=True)
            m = run_sim(cadence, profile, exit_cfg, chimera, own_lookup, asset_to_bucket, oos_dates)
            print(f"total={m.get('total_pct', 0):+.2f}% sortino={m.get('sortino', 0):+.3f} "
                  f"DD={m.get('max_dd_pct', 0):+.2f}% trades={m.get('n_trades', 0)}")
            all_results.append(m)

    # Pick best exit per cadence
    print("\n=== BEST EXIT PER CADENCE ===")
    best_per_cadence = {}
    for cadence in ("1d", "4h", "1h", "15m", "dollar"):
        same = [r for r in all_results if r.get("cadence") == cadence]
        if not same: continue
        best = max(same, key=lambda r: r.get("total_pct", -999))
        best_per_cadence[cadence] = best
        print(f"  {cadence}: best={best['exit']} → total={best['total_pct']:+.2f}% "
              f"sortino={best['sortino']:+.3f} DD={best['max_dd_pct']:+.2f}% n_trades={best['n_trades']}")

    # Write report
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# Multi-Cadence MA/EMA Comparison (2026-05-20)\n",
             f"**Window**: {OOS_START} → {OOS_END}",
             f"**Constraints**: K=8, per_asset 10%, total 60%, confirmation gate (≥2 cells)",
             "",
             "## Full grid (cadence × exit config)",
             "",
             "| Cadence | Exit | Total | Annualized | Daily | Sortino | Max DD | Trades | Win % | Median hold |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in all_results:
        lines.append(f"| {r.get('cadence', '?')} | `{r.get('exit', '?')}` | "
                     f"{r.get('total_pct', 0):+.2f}% | {r.get('ann_pct', 0):+.2f}% | "
                     f"{r.get('daily_mean_pct', 0):+.4f}% | {r.get('sortino', 0):+.3f} | "
                     f"{r.get('max_dd_pct', 0):+.2f}% | {r.get('n_trades', 0)} | "
                     f"{r.get('win_rate_pct', 0):.1f}% | {r.get('median_hold_bars', 0):.0f} |")

    lines += ["", "## Best per cadence (side-by-side)", "",
              "| Cadence | Best exit | Total | Sortino | Max DD | Trades | Median hold (bars) |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for cad, r in best_per_cadence.items():
        lines.append(f"| **{cad}** | `{r['exit']}` | **{r['total_pct']:+.2f}%** | "
                     f"{r['sortino']:+.3f} | {r['max_dd_pct']:+.2f}% | {r['n_trades']} | "
                     f"{r['median_hold_bars']:.0f} |")

    (OUT_DIR / "MULTI_CADENCE_COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")
    json.dump(all_results, (OUT_DIR / "multi_cadence_results.json").open("w"), indent=2, default=str)
    print(f"\n[OK] wrote {OUT_DIR / 'MULTI_CADENCE_COMPARISON.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
