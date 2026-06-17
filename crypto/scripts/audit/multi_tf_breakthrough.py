"""Multi-TF Confluence + Sub-Day Entry Breakthrough Harness (2026-05-20)

Test the hypothesis that MA/EMA capture-rate / per-trade-quality can be lifted
by requiring multi-cadence signal agreement before firing, and timing entry on
a sub-day bar after a daily breakout.

PIPELINE
  1. Per-asset cell mining (TRAIN+VAL only, pre-2024-05-15):
     - SmartCandidateGenerator -> ~65 (fast, slow) pairs per ma_type
     - Empirical decorrelation on this asset's TRAIN closes
     - Score each (cell, cadence) by Sharpe-proxy of fwd-N-bar returns post cross
     - Keep top 3 cells per (asset, cadence)
  2. OOS fire generation (2024-05-16 -> 2025-03-15):
     - For each (asset, cadence) using its top cells, emit all cross-up events
  3. Multi-TF confluence gate:
     - confluence_level = number of distinct cadences with a fire within +/-1 day
     - VARIANTS: 1d-only / 1d+4h / 1d+4h+1h
  4. Sub-day entry overlay (V4): on daily-confluence fire, wait for next 1h bar
     where 1h close > daily fire-bar close (continuation). Skip if no
     confirmation within first 6 hours of next day.
  5. Portfolio simulation (honest 4-bound: best/signal/random/worst).

OUTPUT
  runs/audit/MULTI_TF_BREAKTHROUGH_2026_05_20/
    per_asset_top_cells.parquet
    oos_fires_per_cadence.parquet
    confluence_fires_v{1,2,3,4}.parquet
    sim_results.json
    REPORT.md

INVARIANTS
  - LONG-only, spot, no leverage (project hard constraint)
  - Cell selection: TRAIN+VAL only; OOS reserved for evaluation
  - No forward-return in K-selection signal (only confluence_count + raw strength)
  - All 4 ranking bounds reported (best/signal/random/worst)
  - No emoji in print statements (Windows cp1252)
"""
from __future__ import annotations
import sys
import json
import math
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "pipeline"))
sys.path.insert(0, str(ROOT / "scripts" / "audit"))

from smart_candidate_generator import smart_grid, empirical_decorrelate, generate_raw_candidates

OUT_DIR = ROOT / "runs" / "audit" / "MULTI_TF_BREAKTHROUGH_2026_05_20"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- splits (matches honest_v2_per_asset)
TRAIN_VAL_END = date(2024, 5, 15)
OOS_START = date(2024, 5, 16)
OOS_END = date(2025, 3, 15)

# ---- portfolio params (V2 deploy config)
BET_FRACTION = 0.10
HARD_STOP = -0.04
TRAIL_ARM = 0.10
TRAIL_DROP = 0.05
K_MAX = 12
HOLD_MAX_DAYS = 10
COST = 0.0030  # 30 bps round-trip

# ---- cell mining params
FWD_BARS = 5         # forward window for cell scoring (per-cadence bars)
TOP_K_CELLS = 3      # top N cells per (asset, cadence)
MIN_FIRES = 10       # cell needs >=10 fires on TRAIN+VAL to qualify
CORR_THRESHOLD = 0.85  # decorrelation gate

# ---- universe (subset of u50 with multi-cadence chimera)
U50 = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC",
    "SUI", "NEAR", "APT", "DOT", "HBAR", "ALGO", "AR",
    "PEPE", "SHIB", "FLOKI", "BONK",
    "UNI", "AAVE", "JST", "CRV", "FET",
    "ICP", "FIL", "ETC", "TRX", "XLM", "BCH", "OP", "ARB", "CHZ",
    "ENJ", "ZEC", "DASH", "LDO", "SUPER", "DYDX",
]


# =====================================================================
# 1. CELL MINING
# =====================================================================

def _ma(s: np.ndarray, period: int, ma_type: str) -> np.ndarray:
    sr = pd.Series(s)
    if ma_type == "SMA":
        return sr.rolling(period).mean().values
    return sr.ewm(span=period, adjust=False).mean().values


def cross_up_events(closes: np.ndarray, fast: int, slow: int, ma_type: str) -> np.ndarray:
    """Return boolean array of length len(closes); True at indices where
    fast MA crossed above slow MA at this bar."""
    if len(closes) < slow + 2:
        return np.zeros(len(closes), dtype=bool)
    mf = _ma(closes, fast, ma_type)
    ml = _ma(closes, slow, ma_type)
    cross = np.zeros(len(closes), dtype=bool)
    cross[1:] = (mf[1:] > ml[1:]) & (mf[:-1] <= ml[:-1])
    return cross


def cell_score_on_training(
    closes: np.ndarray, dates: np.ndarray, fast: int, slow: int, ma_type: str,
    train_val_end: date, fwd_bars: int = FWD_BARS,
) -> dict:
    """Score a cell on TRAIN+VAL: mean PnL, hit_rate, Sharpe-proxy."""
    cross = cross_up_events(closes, fast, slow, ma_type)
    # Restrict to TRAIN+VAL events
    n = len(closes)
    fwd_rets = []
    for i in range(n - fwd_bars):
        if not cross[i]:
            continue
        if dates[i] >= train_val_end:
            continue
        ep = closes[i]
        if ep <= 0 or not np.isfinite(ep):
            continue
        # Forward N-bar return
        fp = closes[i + fwd_bars]
        if not np.isfinite(fp) or fp <= 0:
            continue
        fwd_rets.append(fp / ep - 1)
    if len(fwd_rets) < MIN_FIRES:
        return {"n": len(fwd_rets), "mean": 0, "hit": 0, "sharpe": 0}
    arr = np.array(fwd_rets)
    mn = arr.mean()
    sd = arr.std()
    hit = (arr > 0).mean()
    sharpe = mn / sd if sd > 0 else 0
    return {"n": len(arr), "mean": mn, "hit": hit, "sharpe": sharpe}


def mine_top_cells_for_asset(
    chimera_loader, asset: str, cadence: str, candidates: list[tuple[int, int]],
    train_val_end: date, top_k: int = TOP_K_CELLS,
) -> list[dict]:
    """For one asset + cadence, score all candidate cells on both SMA + EMA,
    return top-K by Sharpe-proxy."""
    try:
        df = chimera_loader.load(asset, cadence)
    except Exception as e:
        print(f"  load fail {asset} {cadence}: {e}")
        return []
    if df is None:
        return []
    if hasattr(df, "to_pandas"):
        df = df.to_pandas()
    if "date" not in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
    else:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("timestamp" if "timestamp" in df.columns else "date").reset_index(drop=True)
    closes = df["close"].values.astype(float)
    dates = df["date"].values
    if len(closes) < 200:
        return []

    results = []
    for ma_type in ["SMA", "EMA"]:
        for (f, s) in candidates:
            sc = cell_score_on_training(closes, dates, f, s, ma_type, train_val_end)
            sc.update({"asset": asset, "cadence": cadence, "ma_type": ma_type, "fast": f, "slow": s})
            results.append(sc)
    # Filter to cells with enough fires
    qual = [r for r in results if r["n"] >= MIN_FIRES]
    qual.sort(key=lambda x: x["sharpe"], reverse=True)
    return qual[:top_k]


def mine_all_cells(chimera_loader, universe: list[str], cadences: list[str]) -> pd.DataFrame:
    """For every (asset, cadence), generate decorrelated smart candidates on
    that asset's TRAIN closes, mine top cells. Return long-format DataFrame."""
    rows = []
    raw_cands = generate_raw_candidates(max_period=100)
    print(f"raw smart candidates per asset (pre-decorrelation): {len(raw_cands)}")
    for asset in universe:
        # decorrelate using asset's daily TRAIN closes (cheap reference)
        try:
            df_d = chimera_loader.load(asset, "1d")
            if df_d is None:
                continue
            if hasattr(df_d, "to_pandas"):
                df_d = df_d.to_pandas()
            if "date" not in df_d.columns:
                df_d["date"] = pd.to_datetime(df_d["timestamp"], unit="ms").dt.date
            mask = pd.to_datetime(df_d["date"]) < pd.Timestamp(TRAIN_VAL_END)
            ref_closes = df_d.loc[mask, "close"].values.astype(float)
            if len(ref_closes) < 200:
                cands_for_asset = raw_cands
            else:
                cands_for_asset = empirical_decorrelate(ref_closes, raw_cands, CORR_THRESHOLD, "SMA")
        except Exception:
            cands_for_asset = raw_cands
        for cad in cadences:
            top = mine_top_cells_for_asset(chimera_loader, asset, cad, cands_for_asset, TRAIN_VAL_END)
            for r in top:
                rows.append(r)
        print(f"  mined {asset}: {sum(1 for r in rows if r['asset']==asset)} cells across {len(cadences)} cadences")
    return pd.DataFrame(rows)


# =====================================================================
# 2. OOS FIRE GENERATION
# =====================================================================

def generate_oos_fires(chimera_loader, top_cells_df: pd.DataFrame,
                       oos_start: date, oos_end: date) -> pd.DataFrame:
    """For each (asset, cadence, cell) in top_cells, generate OOS fires.
    Returns long-format with columns: asset, cadence, ma_type, fast, slow,
    fire_ts, fire_date, signal_strength."""
    fires = []
    for asset in top_cells_df["asset"].unique():
        sub = top_cells_df[top_cells_df["asset"] == asset]
        for cadence in sub["cadence"].unique():
            sub_cad = sub[sub["cadence"] == cadence]
            try:
                df = chimera_loader.load(asset, cadence)
                if df is None:
                    continue
                if hasattr(df, "to_pandas"):
                    df = df.to_pandas()
                if "date" not in df.columns:
                    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
                else:
                    df["date"] = pd.to_datetime(df["date"]).dt.date
                df = df.sort_values("timestamp" if "timestamp" in df.columns else "date").reset_index(drop=True)
            except Exception:
                continue
            closes = df["close"].values.astype(float)
            dates = df["date"].values
            ts = df["timestamp"].values if "timestamp" in df.columns else None
            for _, cell in sub_cad.iterrows():
                f, s, mt = int(cell["fast"]), int(cell["slow"]), cell["ma_type"]
                cross = cross_up_events(closes, f, s, mt)
                if not cross.any():
                    continue
                mf = _ma(closes, f, mt)
                ml = _ma(closes, s, mt)
                for i in np.where(cross)[0]:
                    d = dates[i]
                    if d < oos_start or d > oos_end:
                        continue
                    strength = (mf[i] - ml[i]) / closes[i] if closes[i] > 0 else 0
                    fires.append({
                        "asset": asset, "cadence": cadence, "ma_type": mt,
                        "fast": f, "slow": s,
                        "fire_ts": int(ts[i]) if ts is not None else 0,
                        "fire_date": d,
                        "signal_strength": float(strength),
                    })
    return pd.DataFrame(fires)


# =====================================================================
# 3. MULTI-TF CONFLUENCE
# =====================================================================

def apply_confluence_gate(fires_df: pd.DataFrame, required_cadences: set[str],
                          window_days: int = 1) -> pd.DataFrame:
    """Filter fires to those where ALL required cadences have fired
    on this asset within window_days of each other.
    Returns dataframe keyed (asset, fire_date) — using 1d fire as anchor.
    Adds `confluence_level` = number of distinct cadences agreeing.
    """
    # Index fires by (asset, cadence) -> list of dates
    out = []
    if not len(fires_df):
        return pd.DataFrame()
    fires_df = fires_df.copy()
    fires_df["fire_date"] = pd.to_datetime(fires_df["fire_date"]).dt.date
    by_asset_cad = fires_df.groupby(["asset", "cadence"])["fire_date"].apply(set).to_dict()

    for asset in fires_df["asset"].unique():
        # Use 1d fires as anchor (the daily MA cross day)
        d1_dates = sorted(by_asset_cad.get((asset, "1d"), set()))
        for anchor in d1_dates:
            # Check each required cadence has a fire within window
            agreeing = {"1d"}
            for cad in required_cadences:
                if cad == "1d":
                    continue
                cad_dates = by_asset_cad.get((asset, cad), set())
                # any cad fire in [anchor - window_days, anchor + window_days]?
                for delta in range(-window_days, window_days + 1):
                    d = anchor + timedelta(days=delta)
                    if d in cad_dates:
                        agreeing.add(cad)
                        break
            if required_cadences.issubset(agreeing):
                # All required cadences have agreed
                # Pull the 1d row(s) for this anchor as the trade record
                rows = fires_df[(fires_df["asset"] == asset) & (fires_df["cadence"] == "1d") &
                                (fires_df["fire_date"] == anchor)]
                for _, r in rows.iterrows():
                    out.append({
                        "asset": asset,
                        "fire_date": anchor,
                        "ma_type": r["ma_type"], "fast": int(r["fast"]), "slow": int(r["slow"]),
                        "signal_strength": float(r["signal_strength"]),
                        "confluence_level": len(agreeing),
                        "confluence_cadences": ",".join(sorted(agreeing)),
                    })
    return pd.DataFrame(out)


# =====================================================================
# 4. SUB-DAY ENTRY OVERLAY
# =====================================================================

def apply_sub_day_entry(confluence_df: pd.DataFrame, chimera_loader,
                       confirm_hours: int = 6) -> pd.DataFrame:
    """For each (asset, fire_date), look at the next day's first `confirm_hours`
    1h bars. Enter at the first 1h close > the daily close on fire_date.
    Skip if no 1h confirmation within the window.

    Adds: entry_ts, entry_price, entry_lag_hours.
    """
    if not len(confluence_df):
        return confluence_df.assign(entry_ts=pd.NaT, entry_price=np.nan, entry_lag_hours=np.nan)
    out_rows = []
    for asset in confluence_df["asset"].unique():
        try:
            df_d = chimera_loader.load(asset, "1d")
            df_1h = chimera_loader.load(asset, "1h")
            if df_d is None or df_1h is None:
                continue
            for d in [df_d, df_1h]:
                if hasattr(d, "to_pandas"):
                    pass  # will to_pandas below
            df_d = df_d.to_pandas() if hasattr(df_d, "to_pandas") else df_d
            df_1h = df_1h.to_pandas() if hasattr(df_1h, "to_pandas") else df_1h
            df_d["date"] = pd.to_datetime(df_d["timestamp"], unit="ms").dt.date
            df_1h["dt"] = pd.to_datetime(df_1h["timestamp"], unit="ms")
            df_1h = df_1h.sort_values("dt").reset_index(drop=True)
        except Exception:
            continue
        sub = confluence_df[confluence_df["asset"] == asset]
        for _, r in sub.iterrows():
            fd = r["fire_date"]
            daily_row = df_d[df_d["date"] == fd]
            if not len(daily_row):
                continue
            d_close = float(daily_row.iloc[-1]["close"])
            # Next day's first confirm_hours 1h bars
            next_day_start = pd.Timestamp(fd) + pd.Timedelta(days=1)
            next_day_end = next_day_start + pd.Timedelta(hours=confirm_hours)
            window = df_1h[(df_1h["dt"] >= next_day_start) & (df_1h["dt"] < next_day_end)]
            entry_price = np.nan
            entry_ts = pd.NaT
            entry_lag = np.nan
            for i, row in window.iterrows():
                if float(row["close"]) > d_close:
                    entry_price = float(row["close"])
                    entry_ts = row["dt"]
                    entry_lag = (row["dt"] - next_day_start).total_seconds() / 3600.0
                    break
            rd = dict(r)
            rd["entry_ts"] = entry_ts
            rd["entry_price"] = entry_price
            rd["entry_lag_hours"] = entry_lag
            out_rows.append(rd)
    return pd.DataFrame(out_rows)


# =====================================================================
# 5. PORTFOLIO SIMULATION (HONEST 4-BOUND)
# =====================================================================

def _trade_exit(entry_price: float, fwd_closes: list[float],
                stop: float = HARD_STOP, trail_arm: float = TRAIL_ARM,
                trail_drop: float = TRAIL_DROP, hold_max: int = HOLD_MAX_DAYS,
                cost: float = COST) -> tuple[float, int]:
    """Walk forward through daily closes; return (realized_ret, bars_held)."""
    peak = entry_price
    armed = False
    for d, p in enumerate(fwd_closes, start=1):
        if p is None or not np.isfinite(p) or p <= 0:
            break
        r = p / entry_price - 1
        if p > peak:
            peak = p
        if not armed and r >= trail_arm:
            armed = True
        if r <= stop:
            return stop - cost, d
        if armed and p <= peak * (1 - trail_drop):
            return p / entry_price - 1 - cost, d
        if d >= hold_max:
            return r - cost, d
    if not fwd_closes:
        return -cost, 0
    last_valid = next((p for p in reversed(fwd_closes) if p and np.isfinite(p)), None)
    if last_valid is None:
        return -cost, 0
    return last_valid / entry_price - 1 - cost, len(fwd_closes)


def _build_panel_idx(chimera_loader, assets: list[str]) -> dict[str, pd.DataFrame]:
    """For each asset, load 1d chimera as pandas DF with `date` column."""
    out = {}
    for a in assets:
        try:
            df = chimera_loader.load(a, "1d")
            if df is None:
                continue
            if hasattr(df, "to_pandas"):
                df = df.to_pandas()
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
            df = df[["date", "high", "low", "close"]].sort_values("date").reset_index(drop=True)
            out[a] = df
        except Exception:
            continue
    return out


def simulate_4bound(
    fires_df: pd.DataFrame, panel_idx: dict[str, pd.DataFrame],
    oos_start: date, oos_end: date, mode: str = "signal",
    use_sub_day_entry: bool = False, rng_seed: int = 42,
    total_deploy_cap: float = 0.60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """4-bound portfolio simulation.

    mode:
      best    -- rank by forward-5d return (perfect foresight upper bound)
      signal  -- rank by (confluence_level desc, signal_strength_pct desc)
      random  -- uniform random ranker
      worst   -- rank by forward-5d return ASCENDING

    Returns (daily_df, trade_log).
    """
    rng = np.random.default_rng(rng_seed)
    portfolio = 1.0
    cash = 1.0
    open_positions: list[dict] = []
    trades: list[dict] = []
    daily_records: list[dict] = []

    if not len(fires_df):
        return pd.DataFrame(daily_records), pd.DataFrame(trades)

    fires_df = fires_df.copy()
    fires_df["fire_date"] = pd.to_datetime(fires_df["fire_date"]).dt.date

    # Compute fwd-5d return per fire (for best/worst modes); requires panel lookup
    if mode in ("best", "worst"):
        fwd_rets = []
        for _, r in fires_df.iterrows():
            sub = panel_idx.get(r["asset"])
            if sub is None:
                fwd_rets.append(np.nan)
                continue
            sub_oos = sub[sub["date"] >= r["fire_date"]].head(6)
            if len(sub_oos) < 6:
                fwd_rets.append(np.nan)
                continue
            ep = float(sub_oos.iloc[0]["close"])
            fp = float(sub_oos.iloc[5]["close"])
            if ep > 0 and np.isfinite(ep) and np.isfinite(fp) and fp > 0:
                fwd_rets.append(fp / ep - 1)
            else:
                fwd_rets.append(np.nan)
        fires_df["fwd_5d"] = fwd_rets
        fires_df["rank_sig"] = fires_df["fwd_5d"].fillna(-999 if mode == "best" else 999)
    elif mode == "signal":
        # rank by confluence_level desc, then strength_pct (within-day) desc
        if "confluence_level" not in fires_df.columns:
            fires_df["confluence_level"] = 1
        fires_df["strength_pct"] = fires_df.groupby("fire_date")["signal_strength"].rank(pct=True)
        fires_df["rank_sig"] = (fires_df["confluence_level"].astype(float)
                                 + fires_df["strength_pct"].astype(float))
    elif mode == "random":
        fires_df["rank_sig"] = rng.random(len(fires_df))
    else:
        raise ValueError(f"unknown mode {mode}")

    # Sort by fire_date for daily iteration
    fires_by_date = fires_df.groupby("fire_date")

    # Walk the OOS calendar
    cur = oos_start
    while cur <= oos_end:
        # 1) Close positions whose exit_date <= today
        new_open = []
        for pos in open_positions:
            if cur >= pos["exit_date"]:
                sub = panel_idx.get(pos["asset"])
                if sub is None:
                    new_open.append(pos)
                    continue
                fwd_sub = sub[(sub["date"] > pos["entry_date"]) & (sub["date"] <= cur)]
                fwd_closes = [float(p) if np.isfinite(p) else None for p in fwd_sub["close"].values]
                r, d_held = _trade_exit(pos["entry_price"], fwd_closes)
                pnl = pos["bet_size"] * r
                cash += pos["bet_size"] + pnl
                trades.append({
                    "asset": pos["asset"], "entry_date": pos["entry_date"],
                    "exit_date": cur, "days_held": d_held,
                    "bet_size": pos["bet_size"], "realized_ret": r,
                    "ma_type": pos.get("ma_type"), "fast": pos.get("fast"), "slow": pos.get("slow"),
                    "confluence_level": pos.get("confluence_level", 1),
                })
            else:
                new_open.append(pos)
        open_positions = new_open

        # 2) Open new positions from today's fires
        if cur in fires_by_date.groups:
            today = fires_by_date.get_group(cur).copy()
            asc = (mode == "worst")
            today = today.sort_values("rank_sig", ascending=asc)
            open_assets = set(p["asset"] for p in open_positions)
            today = today[~today["asset"].isin(open_assets)]
            today = today.drop_duplicates(subset="asset", keep="first")

            for _, ev in today.iterrows():
                if len(open_positions) >= K_MAX:
                    break
                # Enforce total deploy cap (matches deployed sleeve constraint)
                current_deploy_frac = sum(p["bet_size"] for p in open_positions) / max(portfolio, 1e-9)
                if current_deploy_frac >= total_deploy_cap:
                    break
                bet = BET_FRACTION * portfolio
                if (current_deploy_frac + bet / max(portfolio, 1e-9)) > total_deploy_cap:
                    bet = max((total_deploy_cap - current_deploy_frac) * portfolio, 0.0)
                if bet <= 0:
                    break
                if cash < bet:
                    break
                # Entry price: sub_day_entry uses entry_price from confluence_df if available
                ep = np.nan
                ed = cur
                if use_sub_day_entry and "entry_price" in ev.index and pd.notna(ev.get("entry_price")):
                    ep = float(ev["entry_price"])
                    if pd.notna(ev.get("entry_ts")):
                        ed = pd.Timestamp(ev["entry_ts"]).date()
                if not np.isfinite(ep) or ep <= 0:
                    sub = panel_idx.get(ev["asset"])
                    if sub is None:
                        continue
                    drow = sub[sub["date"] == cur]
                    if not len(drow):
                        continue
                    ep = float(drow.iloc[0]["close"])
                if not np.isfinite(ep) or ep <= 0:
                    continue
                cash -= bet
                open_positions.append({
                    "asset": ev["asset"], "entry_date": ed,
                    "entry_price": ep,
                    "exit_date": cur + timedelta(days=HOLD_MAX_DAYS),
                    "bet_size": bet,
                    "ma_type": ev.get("ma_type"), "fast": ev.get("fast"), "slow": ev.get("slow"),
                    "confluence_level": int(ev.get("confluence_level", 1)),
                })

        # 3) MtM
        omtm = 0
        for pos in open_positions:
            sub = panel_idx.get(pos["asset"])
            if sub is None:
                omtm += pos["bet_size"]
                continue
            av = sub[sub["date"] <= cur]
            if not len(av):
                omtm += pos["bet_size"]
                continue
            cp = float(av.iloc[-1]["close"])
            if not np.isfinite(cp):
                omtm += pos["bet_size"]
                continue
            omtm += pos["bet_size"] * (cp / pos["entry_price"])
        portfolio = cash + omtm
        daily_records.append({"date": cur, "portfolio_value": portfolio, "n_open": len(open_positions)})
        cur += timedelta(days=1)

    return pd.DataFrame(daily_records), pd.DataFrame(trades)


def metrics(daily: pd.DataFrame, trades: pd.DataFrame, window_days: int) -> dict:
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
    if n:
        win = (trades["realized_ret"] > 0).mean() * 100
        aw = trades.loc[trades["realized_ret"] > 0, "realized_ret"].mean() * 100 if (trades["realized_ret"] > 0).any() else 0
        al = trades.loc[trades["realized_ret"] < 0, "realized_ret"].mean() * 100 if (trades["realized_ret"] < 0).any() else 0
        mw = trades["realized_ret"].max() * 100
    else:
        win = aw = al = mw = 0
    # 7d rolling check
    if len(pv) >= 7:
        ret_7d = (pv[7:] / pv[:-7] - 1) * 100
        pct_7d_above_5_25 = (ret_7d >= 5.25).mean() * 100
    else:
        pct_7d_above_5_25 = 0
    return {
        "total_pct": float(total), "ann_pct": float(ann), "sharpe": float(sharpe),
        "sortino": float(sortino), "max_dd_pct": float(max_dd), "calmar": float(calmar),
        "n_trades": int(n), "win_rate_pct": float(win), "avg_win_pct": float(aw),
        "avg_loss_pct": float(al), "max_win_pct": float(mw),
        "pct_days_7d_above_5_25": float(pct_7d_above_5_25),
    }


# =====================================================================
# MAIN
# =====================================================================

def main(stage: str = "all"):
    from pipeline.chimera_loader import ChimeraLoader  # type: ignore
    cl = ChimeraLoader()

    print("=" * 78)
    print("MULTI-TF BREAKTHROUGH HARNESS")
    print(f"  universe: {len(U50)} assets")
    print(f"  cadences: 1d, 4h, 1h")
    print(f"  TRAIN+VAL end: {TRAIN_VAL_END}")
    print(f"  OOS: {OOS_START} -> {OOS_END}")
    print("=" * 78)

    # ---- STAGE 1: cell mining
    cells_path = OUT_DIR / "per_asset_top_cells.parquet"
    if stage in ("all", "mine") or not cells_path.exists():
        print("\n[1/5] Mining top cells per (asset, cadence) on TRAIN+VAL...")
        cells_df = mine_all_cells(cl, U50, ["1d", "4h", "1h"])
        cells_df.to_parquet(cells_path, index=False)
        print(f"  saved {len(cells_df)} (asset, cadence, cell) rows to {cells_path}")
    else:
        cells_df = pd.read_parquet(cells_path)
        print(f"\n[1/5] LOADED {len(cells_df)} top cells from {cells_path}")

    # ---- STAGE 2: OOS fire generation
    fires_path = OUT_DIR / "oos_fires_per_cadence.parquet"
    if stage in ("all", "fires") or not fires_path.exists():
        print("\n[2/5] Generating OOS fires...")
        fires = generate_oos_fires(cl, cells_df, OOS_START, OOS_END)
        fires.to_parquet(fires_path, index=False)
        print(f"  saved {len(fires)} OOS fires to {fires_path}")
    else:
        fires = pd.read_parquet(fires_path)
        print(f"\n[2/5] LOADED {len(fires)} OOS fires")

    # ---- STAGE 3: Confluence variants
    print("\n[3/5] Applying multi-TF confluence gates...")
    v1_fires = apply_confluence_gate(fires, required_cadences={"1d"})
    v2_fires = apply_confluence_gate(fires, required_cadences={"1d", "4h"})
    v3_fires = apply_confluence_gate(fires, required_cadences={"1d", "4h", "1h"})
    v1_fires.to_parquet(OUT_DIR / "confluence_fires_v1.parquet", index=False)
    v2_fires.to_parquet(OUT_DIR / "confluence_fires_v2.parquet", index=False)
    v3_fires.to_parquet(OUT_DIR / "confluence_fires_v3.parquet", index=False)
    print(f"  V1 (1d only): {len(v1_fires)} fires")
    print(f"  V2 (1d+4h):   {len(v2_fires)} fires")
    print(f"  V3 (1d+4h+1h): {len(v3_fires)} fires")

    # ---- STAGE 4: Sub-day entry overlay (apply to V2)
    print("\n[4/5] Applying sub-day entry overlay to V2 fires...")
    v4_fires = apply_sub_day_entry(v2_fires.copy(), cl, confirm_hours=6)
    v4_fires.to_parquet(OUT_DIR / "confluence_fires_v4.parquet", index=False)
    v4_with_entry = v4_fires[v4_fires["entry_price"].notna()] if len(v4_fires) else v4_fires
    print(f"  V4 (V2 + sub-day): {len(v4_fires)} total, {len(v4_with_entry)} with confirmed entry")

    # ---- STAGE 5: 4-bound simulation
    print("\n[5/5] Running 4-bound portfolio simulation per variant...")
    panel_idx = _build_panel_idx(cl, U50)
    print(f"  panel loaded for {len(panel_idx)} assets")

    window = (OOS_END - OOS_START).days

    all_results: dict = {}
    variants = [
        ("V1_1d_only", v1_fires, False),
        ("V2_1d_4h", v2_fires, False),
        ("V3_1d_4h_1h", v3_fires, False),
        ("V4_V2_subday", v4_with_entry, True),
    ]
    for vname, vdf, use_sub in variants:
        if not len(vdf):
            print(f"  {vname}: NO FIRES, skipping")
            continue
        print(f"\n--- {vname} ({len(vdf)} fires) ---")
        var_results = {}
        for mode in ["best", "signal", "random", "worst"]:
            daily, trade_log = simulate_4bound(vdf, panel_idx, OOS_START, OOS_END,
                                                mode=mode, use_sub_day_entry=use_sub)
            m = metrics(daily, trade_log, window)
            var_results[mode] = m
            print(f"  {mode:7s}-K: NAV={m['total_pct']:+8.2f}%  Sortino={m['sortino']:+.2f}  "
                  f"DD={m['max_dd_pct']:+.1f}%  n={m['n_trades']}  win={m['win_rate_pct']:.1f}%  "
                  f"7d>=5.25%: {m['pct_days_7d_above_5_25']:.1f}%")
        all_results[vname] = var_results

    (OUT_DIR / "sim_results.json").write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nSaved results to {OUT_DIR / 'sim_results.json'}")

    # ---- Write report
    write_report(all_results, fires, v1_fires, v2_fires, v3_fires, v4_with_entry)


def write_report(results: dict, fires_all: pd.DataFrame,
                 v1: pd.DataFrame, v2: pd.DataFrame, v3: pd.DataFrame, v4: pd.DataFrame):
    lines = ["# Multi-TF Confluence + Sub-Day Entry Breakthrough Report\n"]
    lines.append(f"\n**Date**: 2026-05-20  \n**Window**: {OOS_START} to {OOS_END}  ")
    lines.append(f"**Universe**: {len(U50)} assets  \n")
    lines.append("\n## Fire counts per variant\n")
    lines.append("| Variant | Definition | Total fires | Unique asset-days |")
    lines.append("|---|---|---:|---:|")
    lines.append(f"| ALL | every cell fire (1d/4h/1h merged) | {len(fires_all)} | - |")
    lines.append(f"| V1 | 1d-only | {len(v1)} | {v1.drop_duplicates(['asset','fire_date']).shape[0] if len(v1) else 0} |")
    lines.append(f"| V2 | 1d AND 4h agree (+/-1d window) | {len(v2)} | {v2.drop_duplicates(['asset','fire_date']).shape[0] if len(v2) else 0} |")
    lines.append(f"| V3 | 1d AND 4h AND 1h agree | {len(v3)} | {v3.drop_duplicates(['asset','fire_date']).shape[0] if len(v3) else 0} |")
    lines.append(f"| V4 | V2 + sub-day 1h confirmed entry | {len(v4)} | {v4.drop_duplicates(['asset','fire_date']).shape[0] if len(v4) else 0} |")

    lines.append("\n## 4-bound portfolio simulation (OOS)\n")
    lines.append("| Variant | Mode | NAV % | Sortino | Max DD % | Trades | Win % | 7d>=5.25% |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for vname in ["V1_1d_only", "V2_1d_4h", "V3_1d_4h_1h", "V4_V2_subday"]:
        if vname not in results:
            continue
        for mode in ["best", "signal", "random", "worst"]:
            m = results[vname].get(mode, {})
            if not m:
                continue
            lines.append(f"| {vname} | {mode} | {m['total_pct']:+.2f} | {m['sortino']:+.2f} | "
                         f"{m['max_dd_pct']:+.2f} | {m['n_trades']} | {m['win_rate_pct']:.1f} | "
                         f"{m['pct_days_7d_above_5_25']:.1f} |")

    lines.append("\n## Honest interpretation\n")
    lines.append("- BASELINE (current deploy, separate run): +91.40% / Sortino +3.65 / DD -13.25%")
    lines.append("- V1 signal-K = sanity check; should approximate baseline if cells match")
    lines.append("- V2/V3 signal-K vs V1 signal-K = confluence gate lift (or loss)")
    lines.append("- V4 vs V2 = sub-day entry overlay lift (or loss)")
    lines.append("- random-K = noise floor; signal-K must beat it to claim ranker value")
    lines.append("- best-K = perfect-foresight upper bound; gap to signal-K = ranker headroom")
    lines.append("- 7d>=5.25% = pct of rolling 7-day windows clearing user's wealth-floor cadence")

    (OUT_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "mine", "fires", "sim"])
    args = ap.parse_args()
    main(args.stage)
