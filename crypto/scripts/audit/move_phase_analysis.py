"""move_phase_analysis.py — pre/during/post-move feature analysis.

QUESTION: what distinguishes the moves we CAPTURED from the moves we MISSED?
What predicts ADVERSE (we fired but it lost) entries?

UNIT: (asset, date) event where forward-1d return >= 5% (a "big mover").
  - CAPTURED if strat had a position in that asset on that date.
  - MISSED if mover existed but strat had no position.

UNIT 2: (asset, entry_date) from strat's trade log.
  - WIN if realized_ret > 0
  - LOSS if realized_ret <= 0

FEATURES PER EVENT:
  PRE-move (info available at entry-day close, BEFORE the next-day move):
    - ret_1d_back, ret_3d_back, ret_7d_back
    - vol_zscore (today's volume vs trailing 20d mean)
    - rv_20d (realized vol 20-day)
    - ma_distance (close vs SMA-20)
    - rsi_14
    - btc_regime (bull/chop/bear/crash)
    - asset_30d_regime (bull/chop/bear/crash from asset_own_regime_panel)
    - sector / bucket
  DURING-move (the move itself):
    - fwd_1d_ret
    - high_to_close_ratio (did move continue to close or peak intraday)
  POST-move (next 1-3 days):
    - fwd_2d_to_4d_ret (forward returns after the move)

OUTPUT:
  runs/audit/MA_EMA_PROFILE_2026_05_20/MOVE_PHASE_ANALYSIS.md
  runs/audit/MA_EMA_PROFILE_2026_05_20/move_events.csv
  runs/audit/MA_EMA_PROFILE_2026_05_20/strat_entry_features.csv
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
OWN_REGIME_PANEL = ROOT / "data" / "processed" / "asset_own_regime_panel.parquet"
PROFILE_PATH = ROOT / "data" / "processed" / "per_asset_ma_ema_profile.parquet"
OUT_DIR = ROOT / "runs" / "audit" / "MA_EMA_PROFILE_2026_05_20"
TRADES_CSV = OUT_DIR / "oos_union_trades.csv"
OUT_MD = OUT_DIR / "MOVE_PHASE_ANALYSIS.md"
OUT_EVENTS = OUT_DIR / "move_events.csv"
OUT_STRAT = OUT_DIR / "strat_entry_features.csv"

OOS_START = _date(2024, 5, 16)
OOS_END = _date(2025, 3, 15)

MOVER_THRESHOLD = 0.05  # 5% next-day return = "big mover"


def load_asset_panel():
    print("Loading per-asset 1d OHLCV...")
    panels = {}
    for f in glob.glob(str(CHIMERA_1D / "*_v51_chimera_1d_*.parquet")):
        sym = Path(f).name.split("_")[0].upper().replace("USDT", "")
        try:
            df = pl.read_parquet(f, columns=["timestamp", "open", "high", "low", "close", "volume"]).to_pandas()
        except Exception:
            continue
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.normalize()
        df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        if len(df) < 50:
            continue
        panels[sym] = df
    print(f"  {len(panels)} assets")
    return panels


def add_features(df):
    """Compute pre-move features per row."""
    df = df.copy()
    c = df["close"]
    h = df["high"]
    df["ret_1d"] = c.pct_change()
    df["ret_3d"] = c.pct_change(3)
    df["ret_7d"] = c.pct_change(7)
    df["ret_1d_fwd"] = c.shift(-1) / c - 1
    df["ret_2d_to_4d_fwd"] = c.shift(-4) / c.shift(-1) - 1
    df["high_1d_fwd"] = h.shift(-1)
    df["close_1d_fwd"] = c.shift(-1)
    # vol z-score: today's volume vs trailing 20d
    v = df["volume"]
    vmean = v.rolling(20).mean()
    vstd = v.rolling(20).std()
    df["vol_zscore"] = (v - vmean) / vstd.replace(0, np.nan)
    # 20d realized vol
    df["rv_20d"] = df["ret_1d"].rolling(20).std()
    # SMA-20 distance
    df["sma_20"] = c.rolling(20).mean()
    df["ma_dist_20"] = (c - df["sma_20"]) / df["sma_20"]
    # Simple RSI-14
    delta = c.diff()
    up = delta.clip(lower=0)
    dn = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_dn = dn.rolling(14).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    return df


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=" * 78)
    print("PRE/DURING/POST MOVE PHASE ANALYSIS")
    print("=" * 78)

    panels = load_asset_panel()

    # Strat trades
    print("\nLoading strat trades log...")
    trades = pd.read_csv(TRADES_CSV)
    trades["entry_date"] = pd.to_datetime(trades["entry_date"]).dt.date
    trades["exit_date"] = pd.to_datetime(trades["exit_date"]).dt.date
    print(f"  {len(trades)} trades")

    # Asset's own regime
    own_regime = None
    if OWN_REGIME_PANEL.exists():
        print("Loading asset own-regime panel...")
        own_regime = pl.read_parquet(OWN_REGIME_PANEL).to_pandas()
        if "date" in own_regime.columns:
            own_regime["date"] = pd.to_datetime(own_regime["date"]).dt.date
        print(f"  {len(own_regime)} rows")

    profile_assets = pl.read_parquet(PROFILE_PATH).to_pandas()["asset"].unique()
    print(f"  profile covers {len(profile_assets)} unique assets")

    # ===== Build mover events =====
    print("\nBuilding mover events (fwd 1d >= +5% in OOS window)...")
    events = []
    for sym, df in panels.items():
        df = add_features(df)
        df["date"] = df["date"].dt.date
        df_oos = df[(df["date"] >= OOS_START) & (df["date"] <= OOS_END)]
        movers = df_oos[df_oos["ret_1d_fwd"] >= MOVER_THRESHOLD]
        for _, r in movers.iterrows():
            high_to_close = (r["high_1d_fwd"] - r["close"]) / r["close"] if pd.notna(r["high_1d_fwd"]) else np.nan
            events.append({
                "asset": sym,
                "date": r["date"],
                "ret_1d_back": r["ret_1d"],
                "ret_3d_back": r["ret_3d"],
                "ret_7d_back": r["ret_7d"],
                "vol_zscore": r["vol_zscore"],
                "rv_20d": r["rv_20d"],
                "ma_dist_20": r["ma_dist_20"],
                "rsi_14": r["rsi_14"],
                "fwd_1d_ret": r["ret_1d_fwd"],
                "high_to_close": high_to_close,
                "fwd_2d_to_4d_ret": r["ret_2d_to_4d_fwd"],
                "in_profile": sym in profile_assets,
            })
    events_df = pd.DataFrame(events)
    print(f"  {len(events_df)} mover events")
    print(f"  unique assets w/ movers: {events_df['asset'].nunique()}")

    # Tag CAPTURED vs MISSED
    # CAPTURED if strat had a position in the asset spanning this date
    strat_holdings = []
    for _, t in trades.iterrows():
        for offset in range((t["exit_date"] - t["entry_date"]).days + 1):
            d = t["entry_date"] + pd.Timedelta(days=offset)
            strat_holdings.append({"asset": t["asset"], "date": d.date() if hasattr(d, "date") else d})
    holdings_set = set((h["asset"], h["date"]) for h in strat_holdings)

    events_df["captured"] = events_df.apply(
        lambda r: (r["asset"], r["date"]) in holdings_set, axis=1
    )
    n_cap = int(events_df["captured"].sum())
    n_miss = len(events_df) - n_cap
    print(f"\n  CAPTURED: {n_cap} ({n_cap/len(events_df)*100:.1f}%)")
    print(f"  MISSED:   {n_miss} ({n_miss/len(events_df)*100:.1f}%)")

    # ===== Compare features: CAPTURED vs MISSED =====
    feat_cols = ["ret_1d_back", "ret_3d_back", "ret_7d_back", "vol_zscore", "rv_20d",
                 "ma_dist_20", "rsi_14", "high_to_close", "fwd_2d_to_4d_ret"]
    print("\n=== Feature distributions: CAPTURED vs MISSED ===")
    print(f"  {'feature':<22}{'CAPT_mean':>12}{'CAPT_med':>12}{'MISS_mean':>12}{'MISS_med':>12}{'diff':>10}")
    cap_df = events_df[events_df["captured"]]
    miss_df = events_df[~events_df["captured"]]
    summary_rows = []
    for f in feat_cols:
        cm = cap_df[f].mean()
        cmd = cap_df[f].median()
        mm = miss_df[f].mean()
        mmd = miss_df[f].median()
        diff = cm - mm
        print(f"  {f:<22}{cm:>+12.4f}{cmd:>+12.4f}{mm:>+12.4f}{mmd:>+12.4f}{diff:>+10.4f}")
        summary_rows.append({"feature": f, "capt_mean": cm, "capt_med": cmd,
                             "miss_mean": mm, "miss_med": mmd, "diff_mean": diff})

    # Profile coverage on missed
    in_profile_miss = miss_df["in_profile"].sum()
    not_in_profile_miss = (~miss_df["in_profile"]).sum()
    print(f"\n  MISSED events:")
    print(f"    in profile:     {in_profile_miss} ({in_profile_miss/len(miss_df)*100:.1f}%)")
    print(f"    NOT in profile: {not_in_profile_miss} ({not_in_profile_miss/len(miss_df)*100:.1f}%) "
          f"<- cannot capture by sleeve design")

    events_df.to_csv(OUT_EVENTS, index=False)
    print(f"\n[OK] wrote {OUT_EVENTS}")

    # ===== Strat-entry analysis =====
    print("\n=== Strat entry feature distributions: WIN vs LOSS ===")
    # For each strat entry, look up pre-move features
    entry_records = []
    for _, t in trades.iterrows():
        sym = t["asset"]
        ed = t["entry_date"]
        if sym not in panels:
            continue
        df = add_features(panels[sym])
        df["date"] = df["date"].dt.date
        row = df[df["date"] == ed]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        entry_records.append({
            "asset": sym,
            "entry_date": ed,
            "realized_ret": t["realized_ret"],
            "regime": t["regime"],
            "source": t["source"],
            "exit_reason": t["exit_reason"],
            "ret_1d_back": row["ret_1d"],
            "ret_3d_back": row["ret_3d"],
            "ret_7d_back": row["ret_7d"],
            "vol_zscore": row["vol_zscore"],
            "rv_20d": row["rv_20d"],
            "ma_dist_20": row["ma_dist_20"],
            "rsi_14": row["rsi_14"],
        })
    entry_df = pd.DataFrame(entry_records)
    print(f"  {len(entry_df)} entries with features")

    entry_df["is_win"] = entry_df["realized_ret"] > 0
    win_df = entry_df[entry_df["is_win"]]
    loss_df = entry_df[~entry_df["is_win"]]
    print(f"  WIN:  {len(win_df)}, mean_ret={win_df['realized_ret'].mean()*100:+.2f}%")
    print(f"  LOSS: {len(loss_df)}, mean_ret={loss_df['realized_ret'].mean()*100:+.2f}%")

    print(f"\n  {'feature':<22}{'WIN_mean':>12}{'WIN_med':>12}{'LOSS_mean':>12}{'LOSS_med':>12}{'diff':>10}")
    entry_feat_cols = ["ret_1d_back", "ret_3d_back", "ret_7d_back", "vol_zscore", "rv_20d",
                       "ma_dist_20", "rsi_14"]
    entry_summary = []
    for f in entry_feat_cols:
        wm = win_df[f].mean()
        wmd = win_df[f].median()
        lm = loss_df[f].mean()
        lmd = loss_df[f].median()
        diff = wm - lm
        print(f"  {f:<22}{wm:>+12.4f}{wmd:>+12.4f}{lm:>+12.4f}{lmd:>+12.4f}{diff:>+10.4f}")
        entry_summary.append({"feature": f, "win_mean": wm, "win_med": wmd,
                              "loss_mean": lm, "loss_med": lmd, "diff_mean": diff})

    entry_df.to_csv(OUT_STRAT, index=False)
    print(f"\n[OK] wrote {OUT_STRAT}")

    # ===== Markdown report =====
    lines = [
        "# Move-Phase Analysis: pre/during/post (2026-05-20)\n",
        f"**Window**: {OOS_START} -> {OOS_END}",
        f"**Mover threshold**: forward 1d return >= +{MOVER_THRESHOLD*100:.0f}%",
        f"**Mover events**: {len(events_df)}",
        f"  - CAPTURED by strat: {n_cap} ({n_cap/len(events_df)*100:.1f}%)",
        f"  - MISSED: {n_miss} ({n_miss/len(events_df)*100:.1f}%)",
        f"  - missed and IN profile (could-have): {in_profile_miss} ({in_profile_miss/len(miss_df)*100:.1f}%)",
        f"  - missed but NOT in profile: {not_in_profile_miss} ({not_in_profile_miss/len(miss_df)*100:.1f}%)",
        "",
        "## A. Feature distributions: CAPTURED vs MISSED movers",
        "",
        "What separates moves we caught from moves we missed?",
        "",
        "| Feature | CAPT mean | CAPT med | MISS mean | MISS med | Δ (capt-miss) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(f"| {r['feature']} | {r['capt_mean']:+.4f} | {r['capt_med']:+.4f} | "
                     f"{r['miss_mean']:+.4f} | {r['miss_med']:+.4f} | {r['diff_mean']:+.4f} |")
    lines.extend([
        "",
        "## B. Strat-entry analysis: WIN vs LOSS at entry-day features",
        "",
        f"- Total strat entries with features: {len(entry_df)}",
        f"- WIN entries (positive realized_ret): {len(win_df)} ({len(win_df)/len(entry_df)*100:.1f}%) — "
        f"mean ret {win_df['realized_ret'].mean()*100:+.2f}%",
        f"- LOSS entries: {len(loss_df)} ({len(loss_df)/len(entry_df)*100:.1f}%) — "
        f"mean ret {loss_df['realized_ret'].mean()*100:+.2f}%",
        "",
        "| Feature | WIN mean | WIN med | LOSS mean | LOSS med | Δ (win-loss) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for r in entry_summary:
        lines.append(f"| {r['feature']} | {r['win_mean']:+.4f} | {r['win_med']:+.4f} | "
                     f"{r['loss_mean']:+.4f} | {r['loss_med']:+.4f} | {r['diff_mean']:+.4f} |")

    # ===== Section C: what to learn =====
    lines.extend([
        "",
        "## C. Interpretation",
        "",
        "### PRE-move (what we should LEARN to predict the next mover)",
        "Features with largest Δ between CAPT and MISS are the strongest selection signals.",
        "If `vol_zscore` Δ is large positive, then high-volume days predict capture.",
        "If `rsi_14` Δ is negative, we tend to catch oversold setups.",
        "",
        "### DURING-move",
        "- `high_to_close` measures whether the move continued through the close or peaked intraday.",
        "- High `high_to_close` on missed days = intraday spikes we did not capture (sub-day cadence opportunity).",
        "",
        "### POST-move",
        "- `fwd_2d_to_4d_ret` measures continuation vs mean-reversion.",
        "- If positive, moves continue (trail exit captures more).",
        "- If negative, moves reverse (we should exit faster on profit).",
        "",
        "## D. Actionable gap-closure",
        "",
        f"- **Not-in-profile missed events**: {not_in_profile_miss} movers we cannot capture by design",
        f"  → expand profile to cover {not_in_profile_miss} more events",
        "- **In-profile MISSED events**: these are events we COULD have caught but did not",
        "  → root-cause: confirmation gate failed, K=8 saturated, or no qualifying signal at entry",
        "",
    ])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {OUT_MD}")


if __name__ == "__main__":
    main()
