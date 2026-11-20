"""V5: Multi-TF Confluence Layered on DEPLOYED Per-Asset Cells (2026-05-20).

V1-V4 used my own smart-grid for cell selection. Those gave +27% signal-K vs
deployed +91% baseline -- much weaker. The gap is because:
  (a) deployed uses MA-ACTIVE STATE (`fast > slow`), not cross-up event;
  (b) deployed uses per-asset profile cells (cousin set + winners + fallback),
      not smart-grid;
  (c) deployed has regime-qualified gating and composite deploy-score ranker.

V5 takes the deployed cells (from per_asset_ma_ema_profile.parquet etc.) and
applies them across THREE cadences (1d/4h/1h) using SAME (ma_type, fast, slow)
parameters. Then requires multi-TF agreement (>=2 cadences with at least one
active cell on the day).

This is the cleanest test: same cell selection, just adds the multi-TF layer.

OUTPUT
  runs/audit/MULTI_TF_BREAKTHROUGH_2026_05_20/
    v5_oos_results.json
    v5_REPORT.md
"""
from __future__ import annotations
import sys
import json
import glob
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "pipeline"))

PROFILE = ROOT / "data" / "processed" / "per_asset_ma_ema_profile.parquet"
WINNERS = ROOT / "data" / "processed" / "per_asset_regime_winners.parquet"
FALLBACK = ROOT / "data" / "processed" / "bucket_dna_fallback.parquet"
OWN_REG = ROOT / "data" / "processed" / "asset_own_regime_panel.parquet"
OUT_DIR = ROOT / "runs" / "audit" / "MULTI_TF_BREAKTHROUGH_2026_05_20"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OOS_START = date(2024, 5, 16)
OOS_END = date(2025, 3, 15)

K_MAX = 8
PER_ASSET_CAP = 0.10
TOTAL_DEPLOY_CAP = 0.60
COST_RT = 0.0030
EXIT_HOLD_MAX = 14
EXIT_STOP = -0.04
EXIT_TRAIL_ARM = 0.05
EXIT_TRAIL_DROP = 0.03

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


def load_inputs():
    profile = pd.read_parquet(PROFILE)
    winners = pd.read_parquet(WINNERS)
    fallback = pd.read_parquet(FALLBACK)
    own_regime = pl.read_parquet(OWN_REG).to_pandas()
    own_regime["date"] = pd.to_datetime(own_regime["date"]).dt.normalize()
    return profile, winners, fallback, own_regime


def load_universe():
    import yaml
    with open(ROOT / "config" / "universes" / "u50.yaml") as f:
        u50 = yaml.safe_load(f)
    with open(ROOT / "config" / "universes" / "u100.yaml") as f:
        u100 = yaml.safe_load(f)
    asset_to_bucket = {}
    for a in u50["assets"]:
        sym = a["symbol"].replace("USDT", "")
        asset_to_bucket[sym] = a.get("dna", "VOLATILE")
    for a in u100.get("extra_assets", []):
        if a.get("status") != "ready":
            continue
        sym = a["symbol"].replace("USDT", "")
        asset_to_bucket[sym] = a.get("dna", "VOLATILE")
    return asset_to_bucket


def compute_ma_active(closes: np.ndarray, ma_type: str, fast: int, slow: int) -> np.ndarray:
    sr = pd.Series(closes)
    if ma_type == "SMA":
        mf = sr.rolling(fast).mean().values
        ml = sr.rolling(slow).mean().values
    else:
        mf = sr.ewm(span=fast, adjust=False).mean().values
        ml = sr.ewm(span=slow, adjust=False).mean().values
    return mf > ml


def walk_forward_exit(entry_price, fwd_closes, stop=EXIT_STOP, trail_arm=EXIT_TRAIL_ARM,
                      trail_drop=EXIT_TRAIL_DROP, hold_max=EXIT_HOLD_MAX, cost=COST_RT):
    peak = entry_price
    armed = False
    for d, p in enumerate(fwd_closes, start=1):
        if p is None or not np.isfinite(p):
            continue
        r = p / entry_price - 1
        if p > peak:
            peak = p
        if not armed and r >= trail_arm:
            armed = True
        if r <= stop:
            return stop - cost, d, "stop"
        if armed and p <= peak * (1 - trail_drop):
            return p / entry_price - 1 - cost, d, "trail"
        if d >= hold_max:
            return r - cost, d, "max_hold"
    last = next((p for p in reversed(fwd_closes) if p is not None and np.isfinite(p)), None)
    if last is None:
        return -cost, 0, "no_data"
    return last / entry_price - 1 - cost, len(fwd_closes), "expire"


def build_cell_active_matrix(asset_to_bucket, profile, winners, fallback,
                             cadence: str, chim_dir: Path) -> dict:
    """For each asset, compute MA-active state per its deployed cells.
    Returns {asset: (chim_df, [(cell_meta, active_bool_array)])}.

    cadence: '1d', '4h', '1h' — the timeframe over which to compute the MA state.
    """
    cousins = profile[profile["is_cousin_set_member"]].copy()
    cousins_by_asset = {a: g.to_dict("records") for a, g in cousins.groupby("asset")}
    winners_by_asset = {a: g.to_dict("records") for a, g in winners.groupby("asset")}
    fallback_by_bucket = {r["bucket"]: r.to_dict() for _, r in fallback.iterrows()}

    asset_cells: dict = {}
    files = sorted(chim_dir.glob(f"*_v51_chimera_{cadence}_*.parquet"))
    for f in files:
        sym = f.name.split("_")[0].upper().replace("USDT", "")
        if sym not in asset_to_bucket:
            continue
        try:
            df = pl.read_parquet(f, columns=["timestamp", "close"]).to_pandas()
        except Exception:
            continue
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.normalize()
        df = df.sort_values("date").reset_index(drop=True)
        closes = df["close"].values.astype(float)
        cells = []
        # regime winners
        for w in winners_by_asset.get(sym, []):
            active = compute_ma_active(closes, w["ma_type"], int(w["fast"]), int(w["slow"]))
            cells.append({
                "source": "regime_winner",
                "regime_qualifies": {w["own_regime"]},
                "ma_type": w["ma_type"], "fast": int(w["fast"]), "slow": int(w["slow"]),
                "sharpe": float(w["sharpe_in_regime"]),
                "tier_priority": 1,
                "active": active,
            })
        # cousins
        for c in cousins_by_asset.get(sym, []):
            tag = c["regime_tag"]
            q = REGIME_QUALIFIES.get(tag, set())
            active = compute_ma_active(closes, c["ma_type"], int(c["fast"]), int(c["slow"]))
            cells.append({
                "source": "cousin",
                "regime_qualifies": q,
                "ma_type": c["ma_type"], "fast": int(c["fast"]), "slow": int(c["slow"]),
                "sharpe": float(c["sharpe"]),
                "regime_tag": tag,
                "tier_priority": 2,
                "active": active,
            })
        # fallback if no profile
        if not cells:
            bucket = asset_to_bucket.get(sym, "VOLATILE")
            fb = fallback_by_bucket.get(bucket)
            if fb is not None:
                active = compute_ma_active(closes, fb["ma_type"], int(fb["fast"]), int(fb["slow"]))
                cells.append({
                    "source": "bucket_fallback",
                    "regime_qualifies": {"bull", "chop"},
                    "ma_type": fb["ma_type"], "fast": int(fb["fast"]), "slow": int(fb["slow"]),
                    "sharpe": float(fb["mean_sharpe_proxy"]),
                    "regime_tag": "BULL_AND_CHOP",
                    "tier_priority": 3,
                    "active": active,
                })
        asset_cells[sym] = (df, cells)
    return asset_cells


def collapse_subday_to_daily(asset_cells_subday: dict) -> dict:
    """For sub-day cadences (4h, 1h), collapse active state to per-day
    using "ANY active bar in the day". Returns {asset: {date: bool_any_cell_active}}.

    Cell-level resolution is preserved per cell — we return a dict of
    asset -> {date -> [cell_meta dicts active that day]}.
    """
    out = {}
    for asset, (chim, cells) in asset_cells_subday.items():
        if "date" not in chim.columns:
            continue
        # chim.date is normalized timestamp; group by date
        if not len(chim):
            continue
        # For each cell, mark per-day = ANY bar active that day
        date_arr = chim["date"].values
        per_day: dict = {}
        for cell in cells:
            active = cell["active"]
            n = min(len(active), len(date_arr))
            if n == 0:
                continue
            df_tmp = pd.DataFrame({
                "date": date_arr[:n],
                "active": active[:n],
            })
            grp = df_tmp.groupby("date")["active"].any()
            for d, v in grp.items():
                if bool(v):
                    per_day.setdefault(pd.Timestamp(d).normalize(), []).append(cell)
        out[asset] = per_day
    return out


def build_per_day_cell_active(asset_cells: dict) -> dict:
    """For 1d cadence: {asset: {date: [cell_meta dicts active that day]}}."""
    out = {}
    for asset, (chim, cells) in asset_cells.items():
        per_day: dict = {}
        date_arr = chim["date"].values
        for cell in cells:
            for i, d in enumerate(date_arr):
                if i < len(cell["active"]) and bool(cell["active"][i]):
                    per_day.setdefault(pd.Timestamp(d).normalize(), []).append(cell)
        out[asset] = per_day
    return out


def simulate_v5(
    own_lookup: dict, asset_to_bucket: dict,
    active_1d: dict, active_4h: dict, active_1h: dict,
    chim_by_asset: dict,
    required_cadences: set[str], mode: str = "signal",
    rng_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """V5 sim: requires required_cadences (subset of {'1d','4h','1h'}) all
    have at least one active cell on the day for the asset to fire.

    mode: best | signal | random | worst
    """
    rng = np.random.default_rng(rng_seed)
    portfolio = 1.0
    cash = 1.0
    open_positions: list[dict] = []
    trades: list[dict] = []
    daily_records: list[dict] = []

    cur = OOS_START
    cal_dates = []
    while cur <= OOS_END:
        cal_dates.append(cur)
        cur += timedelta(days=1)

    # Pre-compute median cell sharpe for quality normalization
    all_sharpes = []
    for asset, per_day in active_1d.items():
        for d, cs in per_day.items():
            for c in cs:
                if c["sharpe"] > 0:
                    all_sharpes.append(c["sharpe"])
    median_sharpe = float(np.median(all_sharpes)) if all_sharpes else 0.1

    for sim_date in cal_dates:
        sim_dt = pd.Timestamp(sim_date).normalize()

        # Close positions
        new_open = []
        for pos in open_positions:
            chim = chim_by_asset.get(pos["asset"])
            if chim is None:
                new_open.append(pos)
                continue
            fwd = chim[(chim["date"] > pos["entry_date"]) & (chim["date"] <= sim_dt)]
            fwd_closes = [float(p) if np.isfinite(p) else None for p in fwd["close"].values]
            if not fwd_closes:
                new_open.append(pos)
                continue
            r, d_held, reason = walk_forward_exit(pos["entry_price"], fwd_closes)
            if reason in ("stop", "trail", "max_hold"):
                pnl = pos["bet_size"] * r
                cash += pos["bet_size"] + pnl
                trades.append({
                    "asset": pos["asset"], "entry_date": pos["entry_date"],
                    "exit_date": sim_date, "days_held": d_held,
                    "bet_size": pos["bet_size"], "realized_ret": r,
                    "exit_reason": reason, "confluence_count": pos.get("confluence_count", 1),
                })
            else:
                new_open.append(pos)
        open_positions = new_open

        open_assets = set(p["asset"] for p in open_positions)
        candidates = []
        for asset in asset_to_bucket.keys():
            if asset in open_assets:
                continue
            cells_active_1d = active_1d.get(asset, {}).get(sim_dt, [])
            cells_active_4h = active_4h.get(asset, {}).get(sim_dt, [])
            cells_active_1h = active_1h.get(asset, {}).get(sim_dt, [])
            # Build confluence set
            cad_active = set()
            if cells_active_1d:
                cad_active.add("1d")
            if cells_active_4h:
                cad_active.add("4h")
            if cells_active_1h:
                cad_active.add("1h")
            if not required_cadences.issubset(cad_active):
                continue
            # Regime gate on 1d cells (use best-priority active 1d cell)
            own_r = own_lookup.get((asset, sim_dt))
            if own_r is None:
                continue
            # Use 1d cells primarily; require regime qualification
            qualifying_cells = [c for c in cells_active_1d if own_r in c["regime_qualifies"]]
            if not qualifying_cells:
                continue
            best_cell = sorted(qualifying_cells, key=lambda c: (c["tier_priority"], -c["sharpe"]))[0]
            bucket = asset_to_bucket.get(asset, "VOLATILE")
            tier_mult = REGIME_TIER_MULT.get(own_r, 0.5)
            vol_mult = BUCKET_VOL_MULT.get(bucket, 1.0)
            quality_mult = float(np.clip(best_cell["sharpe"] / max(median_sharpe, 0.05), 0.7, 1.3))
            deploy_score = best_cell["sharpe"] * tier_mult * vol_mult * quality_mult
            candidates.append({
                "asset": asset, "bucket": bucket, "regime": own_r,
                "cell_sharpe": best_cell["sharpe"],
                "deploy_score": deploy_score,
                "confluence_count": len(cad_active),
            })

        if not candidates:
            # MtM and continue
            portfolio = cash + sum(p["bet_size"] for p in open_positions)  # approx
            daily_records.append({"date": sim_date, "portfolio_value": portfolio,
                                  "n_open": len(open_positions)})
            continue

        cand_df = pd.DataFrame(candidates)
        # Add forward 5d return for best/worst modes
        if mode in ("best", "worst"):
            fwd_rets = []
            for _, r in cand_df.iterrows():
                chim = chim_by_asset.get(r["asset"])
                if chim is None:
                    fwd_rets.append(np.nan)
                    continue
                sub_oos = chim[chim["date"] >= sim_dt].head(6)
                if len(sub_oos) < 6:
                    fwd_rets.append(np.nan)
                    continue
                ep = float(sub_oos.iloc[0]["close"])
                fp = float(sub_oos.iloc[5]["close"])
                if ep > 0 and np.isfinite(ep) and np.isfinite(fp):
                    fwd_rets.append(fp / ep - 1)
                else:
                    fwd_rets.append(np.nan)
            cand_df["fwd_5d"] = fwd_rets
            cand_df["rank_sig"] = cand_df["fwd_5d"].fillna(-999 if mode == "best" else 999)
            asc = (mode == "worst")
            cand_df = cand_df.sort_values("rank_sig", ascending=asc)
        elif mode == "signal":
            cand_df = cand_df.sort_values(["confluence_count", "deploy_score"],
                                          ascending=[False, False])
        elif mode == "random":
            cand_df["rng"] = rng.random(len(cand_df))
            cand_df = cand_df.sort_values("rng")
        else:
            raise ValueError(mode)

        cand_df = cand_df.drop_duplicates(subset="asset", keep="first")

        # Allocate
        for _, c in cand_df.iterrows():
            # K-cap
            if len(open_positions) >= K_MAX:
                break
            # Total deploy cap
            current_deploy = sum(p["bet_size"] / portfolio for p in open_positions)
            if current_deploy >= TOTAL_DEPLOY_CAP:
                break
            bet = PER_ASSET_CAP * portfolio
            # Don't exceed cap by adding this bet
            if (current_deploy + bet / portfolio) > TOTAL_DEPLOY_CAP:
                bet = (TOTAL_DEPLOY_CAP - current_deploy) * portfolio
            if bet <= 0:
                break
            if cash < bet:
                break
            chim = chim_by_asset.get(c["asset"])
            if chim is None:
                continue
            drow = chim[chim["date"] == sim_dt]
            if not len(drow):
                continue
            ep = float(drow.iloc[0]["close"])
            if not np.isfinite(ep) or ep <= 0:
                continue
            cash -= bet
            open_positions.append({
                "asset": c["asset"], "entry_date": sim_dt,
                "entry_price": ep, "bet_size": bet,
                "confluence_count": c["confluence_count"],
            })

        # MtM
        omtm = 0.0
        for p in open_positions:
            chim = chim_by_asset.get(p["asset"])
            if chim is None:
                omtm += p["bet_size"]
                continue
            av = chim[chim["date"] <= sim_dt]
            if not len(av):
                omtm += p["bet_size"]
                continue
            cp = float(av.iloc[-1]["close"])
            if not np.isfinite(cp):
                omtm += p["bet_size"]
                continue
            omtm += p["bet_size"] * (cp / p["entry_price"])
        portfolio = cash + omtm
        daily_records.append({"date": sim_date, "portfolio_value": portfolio,
                              "n_open": len(open_positions)})

    return pd.DataFrame(daily_records), pd.DataFrame(trades)


def metrics(daily, trades, window_days):
    pv = daily["portfolio_value"].values
    if len(pv) < 2:
        return {}
    dr = pv[1:] / pv[:-1] - 1
    total = (pv[-1] / pv[0] - 1) * 100
    ann = ((1 + total / 100) ** (365 / max(window_days, 1)) - 1) * 100
    sd = dr.std()
    sortino = (dr.mean() / dr[dr < 0].std() * np.sqrt(252)) if (dr < 0).sum() and dr[dr < 0].std() > 0 else 0
    sharpe = (dr.mean() / sd * np.sqrt(252)) if sd > 0 else 0
    cum = pv / pv[0]
    cm = np.maximum.accumulate(cum)
    max_dd = ((cum / cm - 1) * 100).min()
    calmar = ann / abs(max_dd) if max_dd != 0 else 0
    n = len(trades)
    win = (trades["realized_ret"] > 0).mean() * 100 if n else 0
    aw = trades.loc[trades["realized_ret"] > 0, "realized_ret"].mean() * 100 if n and (trades["realized_ret"] > 0).any() else 0
    al = trades.loc[trades["realized_ret"] < 0, "realized_ret"].mean() * 100 if n and (trades["realized_ret"] < 0).any() else 0
    mw = trades["realized_ret"].max() * 100 if n else 0
    if len(pv) >= 7:
        ret_7d = (pv[7:] / pv[:-7] - 1) * 100
        p7 = (ret_7d >= 5.25).mean() * 100
    else:
        p7 = 0
    return {
        "total_pct": float(total), "ann_pct": float(ann),
        "sharpe": float(sharpe), "sortino": float(sortino),
        "max_dd_pct": float(max_dd), "calmar": float(calmar),
        "n_trades": int(n), "win_rate_pct": float(win),
        "avg_win_pct": float(aw), "avg_loss_pct": float(al), "max_win_pct": float(mw),
        "pct_7d_above_5_25": float(p7),
    }


def main():
    print("=" * 78)
    print("V5: Multi-TF Confluence on DEPLOYED per-asset cells")
    print(f"  OOS: {OOS_START} -> {OOS_END}")
    print(f"  K={K_MAX}  per_asset={PER_ASSET_CAP*100:.0f}%  total={TOTAL_DEPLOY_CAP*100:.0f}%")
    print(f"  exit: hold<={EXIT_HOLD_MAX}d stop={EXIT_STOP*100:.0f}% trail+{EXIT_TRAIL_ARM*100:.0f}%/-{EXIT_TRAIL_DROP*100:.0f}%")
    print("=" * 78)

    print("\n[1/5] Loading inputs...")
    profile, winners, fallback, own_regime = load_inputs()
    asset_to_bucket = load_universe()
    print(f"  profile cells: {len(profile)}; winners: {len(winners)}; "
          f"fallback buckets: {len(fallback)}; universe: {len(asset_to_bucket)}")

    own_lookup = {(r["asset"], r["date"]): r["asset_own_regime"]
                   for _, r in own_regime.iterrows()}

    print("\n[2/5] Building cell-active matrices per cadence...")
    chim_dir_1d = ROOT / "data" / "processed" / "chimera" / "1d"
    chim_dir_4h = ROOT / "data" / "processed" / "chimera" / "4h"
    chim_dir_1h = ROOT / "data" / "processed" / "chimera" / "1h"
    cells_1d = build_cell_active_matrix(asset_to_bucket, profile, winners, fallback, "1d", chim_dir_1d)
    cells_4h = build_cell_active_matrix(asset_to_bucket, profile, winners, fallback, "4h", chim_dir_4h)
    cells_1h = build_cell_active_matrix(asset_to_bucket, profile, winners, fallback, "1h", chim_dir_1h)
    print(f"  cells: 1d={len(cells_1d)}  4h={len(cells_4h)}  1h={len(cells_1h)} assets")

    print("\n[3/5] Collapsing sub-day cell-active to per-day boolean...")
    active_1d = build_per_day_cell_active(cells_1d)
    active_4h = collapse_subday_to_daily(cells_4h)
    active_1h = collapse_subday_to_daily(cells_1h)

    chim_by_asset = {a: chim for a, (chim, _) in cells_1d.items()}

    print("\n[4/5] Running V5 variants x 4 modes...")
    window = (OOS_END - OOS_START).days
    all_res: dict = {}

    variants = [
        ("V5A_1d_only_state",     {"1d"}),
        ("V5B_1d_AND_4h_state",   {"1d", "4h"}),
        ("V5C_1d_AND_4h_AND_1h",  {"1d", "4h", "1h"}),
    ]
    for vname, req in variants:
        print(f"\n--- {vname} (required: {sorted(req)}) ---")
        var_res = {}
        for mode in ["best", "signal", "random", "worst"]:
            daily, trades = simulate_v5(
                own_lookup, asset_to_bucket,
                active_1d, active_4h, active_1h, chim_by_asset,
                required_cadences=req, mode=mode,
            )
            m = metrics(daily, trades, window)
            var_res[mode] = m
            print(f"  {mode:7s}-K: NAV={m['total_pct']:+8.2f}%  Sortino={m['sortino']:+.2f}  "
                  f"DD={m['max_dd_pct']:+.1f}%  n={m['n_trades']}  win={m['win_rate_pct']:.1f}%  "
                  f"7d>=5.25%: {m['pct_7d_above_5_25']:.1f}%")
        all_res[vname] = var_res

    (OUT_DIR / "v5_oos_results.json").write_text(json.dumps(all_res, indent=2, default=str))

    # Report
    print("\n[5/5] Writing report...")
    lines = ["# V5: Multi-TF Confluence on DEPLOYED Per-Asset Cells\n",
             f"\n**Date**: 2026-05-20  ",
             f"\n**OOS**: {OOS_START} -> {OOS_END}  ",
             f"\n**Cells**: per_asset_ma_ema_profile (cousin + winners + fallback) = 321 cells across ~40 assets  ",
             f"\n**Sub-day cells**: SAME (ma_type, fast, slow) tuples applied to 4h and 1h closes  ",
             "\n## Results (4-bound, OOS)\n",
             "| Variant | Mode | NAV % | Sortino | Max DD | Trades | Win % | 7d>=5.25% |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for vname in ["V5A_1d_only_state", "V5B_1d_AND_4h_state", "V5C_1d_AND_4h_AND_1h"]:
        for mode in ["best", "signal", "random", "worst"]:
            m = all_res[vname][mode]
            lines.append(f"| {vname} | {mode} | {m['total_pct']:+.2f} | {m['sortino']:+.2f} | "
                         f"{m['max_dd_pct']:+.2f} | {m['n_trades']} | {m['win_rate_pct']:.1f} | "
                         f"{m['pct_7d_above_5_25']:.1f} |")
    lines.append("\n## Interpretation\n")
    lines.append("- BASELINE deployed sleeve (1d-only, MA-active state, regime-gated, composite ranker): +91.40% / Sortino 3.65 / DD -13.25%")
    lines.append("- V5A signal-K = should approximate baseline (validates harness)")
    lines.append("- V5B vs V5A signal-K = MULTI-TF (1d+4h) lift or loss")
    lines.append("- V5C vs V5A signal-K = TRIPLE confluence lift or loss")
    lines.append("- random-K beats signal-K = ranker is sub-random on multi-TF set")
    (OUT_DIR / "v5_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {OUT_DIR / 'v5_REPORT.md'}")


if __name__ == "__main__":
    main()
