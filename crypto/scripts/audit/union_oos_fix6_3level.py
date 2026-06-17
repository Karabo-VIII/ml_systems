"""union_oos_fix6_3level.py -- PRODUCTION-SIM PATCH for the 3-level ranker.

Reuses the production cell-loading + walk-forward simulator from
union_oos_fix_variants.py, but replaces the single-criterion ranker
`sort by deploy_score` with a 3-level lexicographic sort:

    sort by (cadences_agree DESC, active_cells_count DESC, deploy_score DESC)

WHY: V6 found this lifts NAV from deployed fix1 +91.40% to +118%. This file
re-runs the lift INSIDE the production simulator path to confirm the gap
is real, not a sim-implementation artifact (RED-TEAM CRITICAL #1).

VARIANTS RUN:
  fix1_repro          -- production fix1 ranker (baseline check)
  fix6_3level_multi_tf-- 3-level ranker, cadences_agree from 1d/4h/1h cells
  fix6_2level_no_mtf  -- 2-level ranker (confluence, deploy_score); ABLATION:
                         what's the lift from confluence-lexicographic alone?
  fix6_const_cadences -- 3-level ranker but cadences_agree = constant; ABLATION
                         confirms whether cadences_agree is doing real work.

OUTPUT:
  runs/audit/MA_EMA_PROFILE_2026_05_20/fix6_results.json
  runs/audit/MA_EMA_PROFILE_2026_05_20/FIX6_REPORT.md
"""
from __future__ import annotations
import sys
import json
import glob
from pathlib import Path
from datetime import date as _date, timedelta

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "audit"))

# Reuse production constants + helpers
from union_oos_fix_variants import (
    PROFILE_PATH, WINNERS_PATH, FALLBACK_PATH, OWN_REGIME_PATH,
    CHIMERA_DIR, OUT_DIR, OOS_START, OOS_END,
    K_MAX_BASE, PER_ASSET_CAP, TOTAL_DEPLOY_CAP, COST_RT,
    EXIT_HOLD_MAX, EXIT_STOP_BASE, EXIT_TRAIL_ARM, EXIT_TRAIL_DROP,
    REGIME_TIER_MULT, BUCKET_VOL_MULT, REGIME_QUALIFIES,
    load_universe, walk_forward_exit, compute_ma, load_chimera_1d,
)


def load_chimera_subday(cadence: str) -> dict:
    """Load OHLCV for given cadence; collapse to per-day {date: any-bar-active}."""
    chim = {}
    for f in glob.glob(str(CHIMERA_DIR / cadence / f"*_v51_chimera_{cadence}_*.parquet")):
        sym = Path(f).name.split("_")[0].upper().replace("USDT", "")
        try:
            df = pl.read_parquet(f, columns=["timestamp", "close"]).to_pandas()
        except Exception:
            continue
        df["dt"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["date"] = df["dt"].dt.normalize()
        df = df.sort_values("dt").reset_index(drop=True)
        chim[sym] = df
    return chim


def build_asset_cells_1d(asset_to_bucket, chimera, profile, winners, fallback):
    """Same as production: build per-asset 1d cell list with MA-active arrays."""
    cousins = profile[profile["is_cousin_set_member"]].copy()
    cousins_by_asset = {a: g.to_dict("records") for a, g in cousins.groupby("asset")}
    winners_by_asset = {a: g.to_dict("records") for a, g in winners.groupby("asset")}
    fallback_by_bucket = {r["bucket"]: r.to_dict() for _, r in fallback.iterrows()}
    asset_cells = {}
    for asset in (set(asset_to_bucket.keys()) & set(chimera.keys())):
        chim = chimera[asset]
        cells = []
        for w in winners_by_asset.get(asset, []):
            mf, ml = compute_ma(chim["close"].values, w["ma_type"], int(w["fast"]), int(w["slow"]))
            cells.append({
                "source": "regime_winner",
                "regime_qualifies": {w["own_regime"]},
                "ma_type": w["ma_type"], "fast": int(w["fast"]), "slow": int(w["slow"]),
                "sharpe": float(w["sharpe_in_regime"]),
                "tier_priority": 1,
                "active": mf > ml,
            })
        for c in cousins_by_asset.get(asset, []):
            mf, ml = compute_ma(chim["close"].values, c["ma_type"], int(c["fast"]), int(c["slow"]))
            cells.append({
                "source": "cousin",
                "regime_qualifies": REGIME_QUALIFIES.get(c["regime_tag"], set()),
                "ma_type": c["ma_type"], "fast": int(c["fast"]), "slow": int(c["slow"]),
                "sharpe": float(c["sharpe"]),
                "tier_priority": 2,
                "active": mf > ml,
            })
        if not cells:
            bucket = asset_to_bucket.get(asset, "VOLATILE")
            fb = fallback_by_bucket.get(bucket)
            if fb is not None:
                mf, ml = compute_ma(chim["close"].values, fb["ma_type"], int(fb["fast"]), int(fb["slow"]))
                cells.append({
                    "source": "bucket_fallback",
                    "regime_qualifies": {"bull", "chop"},
                    "ma_type": fb["ma_type"], "fast": int(fb["fast"]), "slow": int(fb["slow"]),
                    "sharpe": float(fb["mean_sharpe_proxy"]),
                    "tier_priority": 3,
                    "active": mf > ml,
                })
        asset_cells[asset] = (chim, cells)
    return asset_cells


def build_subday_cell_active_per_day(asset_to_bucket, chimera_subday, profile, winners, fallback) -> dict:
    """For each asset, compute MA-active state on sub-day cadence using the
    SAME (ma_type, fast, slow) tuples as the 1d profile cells. Then collapse
    to per-day boolean: {asset: {date: any_cell_active_that_day}}."""
    cousins = profile[profile["is_cousin_set_member"]].copy()
    cousins_by_asset = {a: g.to_dict("records") for a, g in cousins.groupby("asset")}
    winners_by_asset = {a: g.to_dict("records") for a, g in winners.groupby("asset")}
    fallback_by_bucket = {r["bucket"]: r.to_dict() for _, r in fallback.iterrows()}

    out = {}
    for asset in (set(asset_to_bucket.keys()) & set(chimera_subday.keys())):
        chim = chimera_subday[asset]
        if not len(chim):
            continue
        closes = chim["close"].values
        per_day_any_active: dict = {}
        # Build cell list (same params as 1d profile)
        cell_params = []
        for w in winners_by_asset.get(asset, []):
            cell_params.append((w["ma_type"], int(w["fast"]), int(w["slow"])))
        for c in cousins_by_asset.get(asset, []):
            cell_params.append((c["ma_type"], int(c["fast"]), int(c["slow"])))
        if not cell_params:
            bucket = asset_to_bucket.get(asset, "VOLATILE")
            fb = fallback_by_bucket.get(bucket)
            if fb is not None:
                cell_params.append((fb["ma_type"], int(fb["fast"]), int(fb["slow"])))
        if not cell_params:
            continue
        # Compute MA-active per cell on sub-day bars; collapse to per-day
        date_arr = chim["date"].values
        for (mt, fast, slow) in cell_params:
            mf, ml = compute_ma(closes, mt, fast, slow)
            active = mf > ml
            n = min(len(active), len(date_arr))
            if n == 0:
                continue
            df_tmp = pd.DataFrame({"date": date_arr[:n], "active": active[:n]})
            for d, v in df_tmp.groupby("date")["active"].any().items():
                if bool(v):
                    per_day_any_active.setdefault(pd.Timestamp(d).normalize(), True)
        out[asset] = per_day_any_active
    return out


def simulate(
    mode: str,
    asset_to_bucket: dict,
    asset_cells_1d: dict,
    sub_4h_active: dict,
    sub_1h_active: dict,
    own_lookup: dict,
    median_sharpe: float,
    use_3level_ranker: bool = False,
    use_multi_tf_feature: bool = False,
    cadences_agree_constant: int | None = None,
    enforce_signal_flip: bool = True,
    u100_pos_caps: dict | None = None,
    market: str = "futures",
) -> dict:
    """
    ============================================================================
    DEPRECATED 2026-05-21: USE paper_trade_replay_v3 WITH TOGGLES INSTEAD.

    This function is the LEGACY offline simulator. Despite the structural rewrite
    (NAV-fraction math + cost-to-NAV + per-bucket spot fees + u100 pos_caps +
    total_deploy_cap) it STILL drifts 6-12pp from v3 on some indicators because
    the cell-pool source differs (this fn builds fresh from INDICATOR_REGISTRY;
    v3 reads the canonical profile parquet via generic_indicator_sleeve).

    The CANONICAL "same language" path is:
        paper_trade_replay_v3.replay(blend=X, market='spot',
                                       disable_pos_caps=True,
                                       disable_signal_flip=True)
    This produces the offline-equivalent number from the SAME engine that
    produces the deploy-gate number. Zero structural drift possible.

    Keep this function for backward compat with the apples-to-apples sweep at
    scripts/research/fix6_apples_to_apples_per_indicator.py. New work should
    go through v3 toggles. See runs/audit/V3_9_SLEEVES_LEGACY_SPOT_2026_05_21/.
    ============================================================================
    """
    """
    use_3level_ranker: if True, sort by (cadences_agree, confluence, deploy_score).
                       If False, sort by deploy_score (production fix1 behavior).
    use_multi_tf_feature: if True, cadences_agree from 1d/4h/1h.
                          If False, cadences_agree = 1 always.
    cadences_agree_constant: if set, force cadences_agree to this constant for ABLATION.

    2026-05-21 v3/offline parity additions:
    enforce_signal_flip: if True, exits include 'signal_flip' (matches v3 default).
                         When cell.active flips to False after entry, position exits
                         via walk_forward_exit's signal_flip mechanic. Default True.
                         Set False for legacy (no-signal_flip) ablation.
    u100_pos_caps: optional dict {asset_root_upper: max_pct_fraction}. When set,
                   per-asset intent size is clamped to min(PER_ASSET_CAP, u100_pos_caps[asset]).
                   Mirrors v3's u100 pos_cap enforcement (the dominant source of
                   offline-vs-v3 NAV inflation -- 10% intent often clamped to 2-6%).
                   Pass None or {} to disable (legacy behavior).
    """
    # 2026-05-21 STRUCTURAL REWRITE -- NAV-fraction math matching v3.
    # PREVIOUSLY: dollar-bet accounting via portfolio_value + available_cash + bet_size.
    # NOW:        NAV starts at 1.0, positions stored as size_pct (fraction of NAV),
    #             daily MtM accumulates day_pnl_frac, NAV updates by *=(1+day_pnl_frac).
    # This matches paper_trade_replay_v3.replay() arithmetic exactly. Cost is charged
    # to NAV at exit (also matching v3 2026-05-21 cost-fix).
    if enforce_signal_flip:
        _exits_allowed = {"stop_loss", "trail_stop", "max_hold", "signal_flip"}
    else:
        _exits_allowed = {"stop_loss", "trail_stop", "max_hold"}

    # 2026-05-21 PER-MARKET FEE SCHEDULE (matches paper_trade_replay_v3.py:56-87).
    # Identical bucket cost tables so offline + v3 charge IDENTICAL costs per
    # (asset, market) on exit. Removed the legacy flat COST_RT=0.0030 which
    # was halfway between spot (28-44bp) and futures (10-18bp).
    _COST_FUTURES_TAKER = {"BLUE": 0.0018, "STEADY": 0.0012, "VOLATILE": 0.0013, "DEGEN": 0.0010}
    _COST_SPOT_TAKER    = {"BLUE": 0.0028, "STEADY": 0.0032, "VOLATILE": 0.0036, "DEGEN": 0.0044}

    def _bucket_root(asset_root: str) -> str:
        a = str(asset_root).replace("USDT", "")
        if a in ("BTC", "ETH"): return "BLUE"
        if a in ("SOL", "XRP", "BNB", "TRX", "ADA", "LTC", "BCH", "TON",
                 "ALGO", "ETC", "ATOM", "AVAX", "LINK", "DOT"): return "STEADY"
        if a in ("ZEC", "PEPE", "WLD", "DASH", "FIL", "FET", "BONK", "JST",
                 "FLOKI", "BLUR", "SHIB", "ORDI", "TRUMP"): return "DEGEN"
        return "VOLATILE"

    def _cost_rt_for(asset: str) -> float:
        bk = _bucket_root(asset)
        if market == "spot":
            return _COST_SPOT_TAKER[bk]
        return _COST_FUTURES_TAKER[bk]

    if market not in ("spot", "futures"):
        raise ValueError(f"market must be 'spot' or 'futures'; got {market!r}")

    nav = 1.0
    open_positions: list[dict] = []   # each: {asset, entry_date, entry_idx, entry_price,
                                       # size_pct, last_mark_price, peak_price, armed,
                                       # entry_cell_active, days_held}
    trade_log = []
    daily_records = []

    cur = OOS_START
    cal_dates = []
    while cur <= OOS_END:
        cal_dates.append(cur)
        cur += timedelta(days=1)

    for sim_date in cal_dates:
        sim_dt = pd.Timestamp(sim_date)
        day_pnl_frac = 0.0
        positions_to_close: list[tuple] = []  # (pos, reason, exit_px)

        # ---- step 1: mark existing positions (daily MtM + exit triggers) ----
        for pos in open_positions:
            chim, _ = asset_cells_1d.get(pos["asset"], (None, []))
            if chim is None:
                continue
            today_row = chim[chim["date"] == sim_dt]
            if today_row.empty:
                # No price for today; skip MtM (NAV unchanged for this position today)
                continue
            today_close = float(today_row.iloc[0]["close"])
            if not np.isfinite(today_close) or today_close <= 0:
                continue
            # Daily MtM (NAV-fraction): position contributes size_pct * day_ret
            day_ret = today_close / pos["last_mark_price"] - 1.0
            day_pnl_frac += pos["size_pct"] * day_ret
            entry_ret = today_close / pos["entry_price"] - 1.0
            # peak / armed for trail
            if today_close > pos["peak_price"]:
                pos["peak_price"] = today_close
            if not pos["armed"] and entry_ret >= EXIT_TRAIL_ARM:
                pos["armed"] = True
            # update mark + days_held
            pos["last_mark_price"] = today_close
            pos["days_held"] = pos.get("days_held", 0) + 1
            # exit triggers (price-based)
            reason = None
            if "stop_loss" in _exits_allowed and entry_ret <= EXIT_STOP_BASE:
                reason = "stop"
            elif pos["armed"] and "trail_stop" in _exits_allowed \
                    and today_close <= pos["peak_price"] * (1 - EXIT_TRAIL_DROP):
                reason = "trail"
            elif "max_hold" in _exits_allowed and pos["days_held"] >= EXIT_HOLD_MAX:
                reason = "max_hold"
            # signal_flip (cell goes inactive at today's chim index)
            if reason is None and "signal_flip" in _exits_allowed \
                    and pos.get("entry_cell_active") is not None:
                today_idx_arr = np.where(chim["date"].values == sim_dt.to_numpy())[0]
                if len(today_idx_arr) > 0:
                    today_idx = int(today_idx_arr[0])
                    act = pos["entry_cell_active"]
                    if today_idx < len(act) and not bool(act[today_idx]):
                        reason = "signal_flip"
            if reason:
                positions_to_close.append((pos, reason, today_close))

        # ---- step 2: realize closes + charge round-trip cost to NAV ----
        for pos, reason, exit_px in positions_to_close:
            pos_ret = exit_px / pos["entry_price"] - 1.0
            # v3-aligned: per-asset, per-market cost charged to NAV via day_pnl_frac.
            cost_rt = _cost_rt_for(pos["asset"])
            day_pnl_frac -= cost_rt * pos["size_pct"]
            trade_log.append({
                "asset": pos["asset"],
                "entry_date": pos["entry_date"],
                "exit_date": sim_date,
                "days_held": pos.get("days_held", 0),
                "bet_size": pos["size_pct"],   # NAV-fraction (legacy schema name)
                "size_pct": pos["size_pct"],
                "gross_ret": pos_ret,
                "cost_rt": cost_rt,
                "realized_ret": pos_ret - cost_rt,  # legacy name
                "exit_reason": reason,
            })
        open_positions = [p for p in open_positions
                          if not any(p is x[0] for x in positions_to_close)]

        open_assets = set(p["asset"] for p in open_positions)
        candidates = []
        for asset, (chim, cells) in asset_cells_1d.items():
            if asset in open_assets:
                continue
            date_mask = chim["date"] == sim_dt
            if not date_mask.any():
                continue
            idx = int(np.where(date_mask)[0][0])
            own_r = own_lookup.get((asset, sim_dt))
            if own_r is None:
                continue
            qualifying_cells = []
            for cell in sorted(cells, key=lambda c: c["tier_priority"]):
                if own_r not in cell["regime_qualifies"]:
                    continue
                if idx >= len(cell["active"]):
                    continue
                if not cell["active"][idx]:
                    continue
                qualifying_cells.append(cell)
            if not qualifying_cells:
                continue
            # CONFIRMATION GATE (same as fix1)
            if len(qualifying_cells) < 2:
                continue
            confluence = len(qualifying_cells)
            best_cell = qualifying_cells[0]
            bucket = asset_to_bucket.get(asset, "VOLATILE")
            tier_mult = REGIME_TIER_MULT.get(own_r, 0.5)
            vol_mult = BUCKET_VOL_MULT.get(bucket, 1.0)
            quality_mult = float(np.clip(best_cell["sharpe"] / max(median_sharpe, 0.05), 0.7, 1.3))
            deploy_score = (best_cell["sharpe"] * tier_mult * vol_mult * quality_mult
                            * (1 + 0.1 * (confluence - 1)))
            # Multi-TF agreement count
            if cadences_agree_constant is not None:
                cad_agree = cadences_agree_constant
            elif use_multi_tf_feature:
                cad_agree = 1
                if sub_4h_active.get(asset, {}).get(sim_dt):
                    cad_agree += 1
                if sub_1h_active.get(asset, {}).get(sim_dt):
                    cad_agree += 1
            else:
                cad_agree = 1
            candidates.append({
                "asset": asset, "bucket": bucket, "regime": own_r,
                "source": best_cell["source"], "cell_sharpe": best_cell["sharpe"],
                "deploy_score": deploy_score, "confluence": confluence,
                "cadences_agree": cad_agree,
                "tier_mult": tier_mult, "vol_mult": vol_mult, "quality_mult": quality_mult,
                "chim": chim,
                # 2026-05-21 PARITY: capture best_cell.active + entry idx so the
                # exit loop can signal_flip when this cell stops firing.
                "best_cell_active": best_cell["active"],
                "entry_idx": idx,
            })

        # ---- step 3: open new positions (NAV-fraction sizing) ----
        if candidates:
            if use_3level_ranker:
                candidates.sort(key=lambda x: (-x["cadences_agree"], -x["confluence"], -x["deploy_score"]))
            else:
                candidates.sort(key=lambda x: -x["deploy_score"])
            slots_remaining = K_MAX_BASE - len(open_positions)
            # Budget in NAV-FRACTION terms (not dollars). Matches v3.
            used_size_frac = sum(p["size_pct"] for p in open_positions)
            budget_frac_remaining = TOTAL_DEPLOY_CAP - used_size_frac
            for cand in candidates:
                if slots_remaining <= 0:
                    break
                if budget_frac_remaining <= 0.001:
                    break
                # Sizing as NAV-fraction (independent of current NAV level).
                # This matches v3's intent.size_pct convention.
                base_frac = TOTAL_DEPLOY_CAP / K_MAX_BASE
                target = base_frac * cand["tier_mult"] * cand["vol_mult"] * cand["quality_mult"]
                hard_cap = PER_ASSET_CAP
                # 2026-05-21 PARITY: u100 pos_cap clamp (per-asset NAV-fraction cap).
                u100_cap_pct = None
                if u100_pos_caps:
                    u100_cap_pct = u100_pos_caps.get(str(cand["asset"]).upper())
                if u100_cap_pct is not None:
                    hard_cap = min(hard_cap, float(u100_cap_pct))
                actual_size_pct = min(target, hard_cap, budget_frac_remaining)
                if actual_size_pct < 0.005:
                    continue
                today_row = cand["chim"][cand["chim"]["date"] == sim_dt]
                if today_row.empty:
                    continue
                ep = float(today_row.iloc[0]["close"])
                if ep <= 0 or not np.isfinite(ep):
                    continue
                budget_frac_remaining -= actual_size_pct
                slots_remaining -= 1
                open_positions.append({
                    "asset": cand["asset"],
                    "entry_date": sim_dt,
                    "entry_idx": cand["entry_idx"],
                    "entry_price": ep,
                    "size_pct": actual_size_pct,
                    "last_mark_price": ep,
                    "peak_price": ep,
                    "armed": False,
                    "days_held": 0,
                    "entry_cell_active": cand["best_cell_active"],
                    # legacy key for compat with any caller inspecting bet_size:
                    "bet_size": actual_size_pct,
                })

        # ---- step 4: NAV update (v3-aligned) ----
        nav *= (1.0 + day_pnl_frac)
        daily_records.append({
            "date": sim_date,
            "portfolio_value": nav,
            "n_open": len(open_positions),
            "day_pnl_frac": day_pnl_frac,
        })

    daily_df = pd.DataFrame(daily_records)
    trades_df = pd.DataFrame(trade_log)
    pv = daily_df["portfolio_value"].values
    if len(pv) < 2:
        return {}
    total_pct = float((pv[-1] / pv[0] - 1) * 100)
    window_days = (OOS_END - OOS_START).days
    ann_pct = float(((1 + total_pct / 100) ** (365 / max(window_days, 1)) - 1) * 100)
    drs = pv[1:] / pv[:-1] - 1
    sortino = float((drs.mean() / drs[drs < 0].std() * np.sqrt(252)) if (drs < 0).sum() and drs[drs < 0].std() > 0 else 0)
    sharpe = float((drs.mean() / drs.std() * np.sqrt(252)) if drs.std() > 0 else 0)
    cum = pv / pv[0]
    cm = np.maximum.accumulate(cum)
    max_dd = float(((cum / cm - 1) * 100).min())
    calmar = float(ann_pct / abs(max_dd)) if max_dd != 0 else 0
    win = float((trades_df["realized_ret"] > 0).mean() * 100) if len(trades_df) else 0
    if len(pv) >= 7:
        r7 = (pv[7:] / pv[:-7] - 1) * 100
        p7 = float((r7 >= 5.25).mean() * 100)
    else:
        p7 = 0
    return {
        "mode": mode,
        "total_pct": total_pct, "ann_pct": ann_pct,
        "sortino": sortino, "sharpe": sharpe,
        "max_dd_pct": max_dd, "calmar": calmar,
        "n_trades": int(len(trades_df)), "win_rate_pct": win,
        "pct_7d_above_5_25": p7,
    }


def main():
    print("=" * 78)
    print("PRODUCTION-SIM PATCH: 3-level ranker + ablations")
    print(f"  OOS {OOS_START} -> {OOS_END}")
    print("=" * 78)

    print("\n[1/3] Loading inputs (production paths)...")
    profile = pd.read_parquet(PROFILE_PATH)
    winners = pd.read_parquet(WINNERS_PATH)
    fallback = pd.read_parquet(FALLBACK_PATH)
    own_regime = pl.read_parquet(OWN_REGIME_PATH).to_pandas()
    own_regime["date"] = pd.to_datetime(own_regime["date"]).dt.normalize()
    asset_to_bucket = load_universe()
    own_lookup = {(r["asset"], r["date"]): r["asset_own_regime"]
                   for _, r in own_regime.iterrows()}
    chimera_1d = load_chimera_1d()
    print(f"  chimera 1d: {len(chimera_1d)} assets")

    asset_cells_1d = build_asset_cells_1d(asset_to_bucket, chimera_1d, profile, winners, fallback)
    print(f"  built 1d cells for {len(asset_cells_1d)} assets")

    median_sharpe = float(np.median([c["sharpe"] for _, (_, cs) in asset_cells_1d.items()
                                       for c in cs if c["sharpe"] > 0])) or 0.1
    print(f"  median cell sharpe: {median_sharpe:.4f}")

    print("\n  loading 4h chimera + computing MA-active per-day collapse...")
    chimera_4h = load_chimera_subday("4h")
    sub_4h_active = build_subday_cell_active_per_day(asset_to_bucket, chimera_4h, profile, winners, fallback)
    print(f"  4h active maps: {len(sub_4h_active)} assets")
    print("  loading 1h chimera + computing MA-active per-day collapse...")
    chimera_1h = load_chimera_subday("1h")
    sub_1h_active = build_subday_cell_active_per_day(asset_to_bucket, chimera_1h, profile, winners, fallback)
    print(f"  1h active maps: {len(sub_1h_active)} assets")

    print("\n[2/3] Running variants...")
    results: dict = {}
    variants = [
        ("fix1_repro",            False, False, None),
        ("fix6_3level_multi_tf",  True,  True,  None),
        ("fix6_2level_no_mtf",    True,  False, None),
        ("fix6_const_cadences",   True,  True,  2),
    ]
    for name, use3, useMTF, const in variants:
        r = simulate(name, asset_to_bucket, asset_cells_1d, sub_4h_active, sub_1h_active,
                     own_lookup, median_sharpe,
                     use_3level_ranker=use3, use_multi_tf_feature=useMTF,
                     cadences_agree_constant=const)
        results[name] = r
        print(f"  {name:25s}: NAV={r['total_pct']:+8.2f}%  Sortino={r['sortino']:+.2f}  "
              f"DD={r['max_dd_pct']:+.1f}%  trades={r['n_trades']}  win={r['win_rate_pct']:.1f}%  "
              f"7d>=5.25%: {r['pct_7d_above_5_25']:.1f}%")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "fix6_results.json").write_text(json.dumps(results, indent=2))

    print("\n[3/3] Writing report...")
    lines = ["# FIX6 PRODUCTION PATCH RESULTS\n",
             "\n## Comparison vs published fix1 (+91.40% / Sortino 3.65 / DD -13.25%)\n",
             "\n| Variant | NAV % | Sortino | DD | Trades | Win % | 7d>=5.25% | Notes |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    notes = {
        "fix1_repro": "Repro of production fix1 ranker (sort by deploy_score)",
        "fix6_3level_multi_tf": "3-level ranker: (cadences_agree, confluence, deploy_score)",
        "fix6_2level_no_mtf": "2-level ranker (confluence, deploy_score); ABLATION on multi-TF",
        "fix6_const_cadences": "cadences_agree = constant 2; ABLATION confirms feature value",
    }
    for n in ["fix1_repro", "fix6_3level_multi_tf", "fix6_2level_no_mtf", "fix6_const_cadences"]:
        r = results[n]
        lines.append(f"| {n} | {r['total_pct']:+.2f} | {r['sortino']:+.2f} | "
                     f"{r['max_dd_pct']:+.2f} | {r['n_trades']} | {r['win_rate_pct']:.1f} | "
                     f"{r['pct_7d_above_5_25']:.1f} | {notes[n]} |")
    lines.append("\n## Interpretation\n")
    lines.append(f"- fix1_repro vs published fix1 (+91.40%): parity check.")
    lines.append("- fix6_3level_multi_tf vs fix1_repro: full V6 lift in production-sim path.")
    lines.append("- fix6_2level_no_mtf vs fix6_3level_multi_tf: contribution of multi-TF (cadences_agree).")
    lines.append("- fix6_const_cadences vs fix6_3level_multi_tf: ablation. If equal -> cadences_agree is doing nothing useful; if much lower -> feature is load-bearing.")
    (OUT_DIR / "FIX6_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {OUT_DIR / 'FIX6_REPORT.md'}")


if __name__ == "__main__":
    main()
