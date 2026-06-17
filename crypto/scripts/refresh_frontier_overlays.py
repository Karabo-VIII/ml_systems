"""Extend the canonical frontier_stable_flow + frontier_etf_flow sleeves forward.

Each canonical pt_* seed was produced from a specific sweep variant:
    pt_frontier_stable_flow  = usdt_shock_K10_H5_mom_select (stable_flow_overlay)
    pt_frontier_etf_flow     = ethbtc_duo_shock_K2_H2        (etf_flow_overlay)

This helper re-runs those two simulations, slices to TEST_START=2025-01-01,
and overwrites the canonical snapshot files. Uses PT_TEST_END (default
2099-12-31) for forward extension.

Run after data refresh so both feature files are current.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TEST_START_CANON = "2025-01-01"
TEST_END = os.environ.get("PT_TEST_END", "2099-12-31")
CAPITAL = 10000.0


def run_stable_flow():
    """Re-simulate the canonical stable-flow champion (usdt_shock K=10 H=5 mom)."""
    from strategy.gen4_frontier_alpha.stable_flow_overlay import (
        build_panel, simulate_overlay, to_snapshot_csv, FEATS
    )

    print("[stable_flow] building panel...", flush=True)
    panel = build_panel(universe_size=50)
    feats = pd.read_parquet(FEATS)
    feats["date"] = pd.to_datetime(feats["date"])
    print(f"[stable_flow] panel {panel.shape}, feats {feats.shape}")
    print(f"[stable_flow] feats date range: {feats['date'].min().date()} -> {feats['date'].max().date()}")

    # Canonical variant: usdt_shock with K=10 H=5 mom_select=True
    print("[stable_flow] simulating canonical variant usdt_shock_K10_H5_mom...")
    eq_df, summary = simulate_overlay(
        panel, feats, signal_col="usdt_shock", K=10, hold_days=5, mom_select=True,
        label="usdt_shock_K10_H5_mom",
    )
    print(f"[stable_flow] SIM: trades={summary.get('n_trades','?')} "
          f"ret={summary.get('total_ret_pct','?'):+.2f}% "
          f"Sh={summary.get('sharpe','?'):.2f} DD={summary.get('max_dd_pct','?'):.1f}%")
    print(f"[stable_flow] final eq ${eq_df['equity'].iloc[-1]:.2f}  "
          f"last_date={eq_df['date'].iloc[-1].date()}")

    # Slice to canonical start (2025-01-01) and rebase equity to CAPITAL at that date
    eq_df["date"] = pd.to_datetime(eq_df["date"])
    sub = eq_df[eq_df["date"] >= TEST_START_CANON].copy().reset_index(drop=True)
    if len(sub) < 2:
        print("[stable_flow] ERROR: no rows after slice")
        return
    start_eq = sub["equity"].iloc[0]
    sub["equity"] = sub["equity"] / start_eq * CAPITAL

    # Write to canonical pt seed
    out = ROOT / "logs" / "paper_trader_v2" / "seeds" / "pt_frontier_stable_flow" / "daily_snapshot.csv"
    to_snapshot_csv(sub, out)
    print(f"[stable_flow] wrote {len(sub)} rows to {out}")
    print(f"[stable_flow] window: {sub['date'].iloc[0].date()} -> {sub['date'].iloc[-1].date()}")
    print(f"[stable_flow] start $  = ${sub['equity'].iloc[0]:.2f}")
    print(f"[stable_flow] end $    = ${sub['equity'].iloc[-1].iloc[0] if isinstance(sub['equity'].iloc[-1], pd.Series) else sub['equity'].iloc[-1]:.2f}")


def run_etf_flow():
    """Re-simulate the canonical etf-flow champion (ethbtc_duo_shock K=2 H=2)."""
    from strategy.gen4_frontier_alpha.etf_flow_overlay import (
        build_panel, simulate, to_snapshot_csv, FEATS
    )

    print("\n[etf_flow] building panel...", flush=True)
    panel = build_panel()
    feats = pd.read_parquet(FEATS)
    feats["date"] = pd.to_datetime(feats["date"])
    print(f"[etf_flow] panel {panel.shape}, feats {feats.shape}")
    print(f"[etf_flow] feats date range: {feats['date'].min().date()} -> {feats['date'].max().date()}")

    # Canonical: ethbtc_duo_shock K=2 H=2
    print("[etf_flow] simulating canonical variant ethbtc_duo_shock_K2_H2...")
    eq_df, summary = simulate(
        panel, feats, mode="ethbtc_duo_shock", K=2, hold_days=2,
        label="ethbtc_duo_shock_K2_H2",
    )
    if "error" in summary:
        print(f"[etf_flow] SIM FAILED: {summary}")
        return
    print(f"[etf_flow] SIM: trades={summary.get('n_trades','?')} "
          f"ret={summary.get('total_ret_pct','?'):+.2f}% "
          f"Sh={summary.get('sharpe','?'):.2f} DD={summary.get('max_dd_pct','?'):.1f}%")

    eq_df["date"] = pd.to_datetime(eq_df["date"])
    sub = eq_df[eq_df["date"] >= TEST_START_CANON].copy().reset_index(drop=True)
    if len(sub) < 2:
        print("[etf_flow] ERROR: no rows after slice")
        return
    start_eq = sub["equity"].iloc[0]
    sub["equity"] = sub["equity"] / start_eq * CAPITAL

    out = ROOT / "logs" / "paper_trader_v2" / "seeds" / "pt_frontier_etf_flow" / "daily_snapshot.csv"
    to_snapshot_csv(sub, out)
    print(f"[etf_flow] wrote {len(sub)} rows to {out}")
    print(f"[etf_flow] window: {sub['date'].iloc[0].date()} -> {sub['date'].iloc[-1].date()}")
    print(f"[etf_flow] end $ = ${sub['equity'].iloc[-1]:.2f}")


if __name__ == "__main__":
    run_stable_flow()
    run_etf_flow()
