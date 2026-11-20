"""Alpha turn-007: A11 Futures-Data Sizing Gate prototype.

Reframes A3/B7 funding-carry-as-trade (which would require short-perp = trades
futures) as a sizing-gate SIGNAL on the spot stack. No futures positions; just
use funding rate as a regime indicator.

Hypothesis:
  - High funding z (retail over-long) -> de-risk spot allocation (fade crowd)
  - Low / negative funding z (shorts crowded) -> full or slightly higher spot
  - Normal -> 1.0

Respects user's hard constraint: SPOT-only, NO leverage, NO futures positions
directionally. Funding data is EXPLOITED as a SIGNAL only.

Inputs:
  - data/frontier/funding/funding_panel_daily.parquet  (daily funding, 2020+)
  - logs/portfolio_aggregator/recommended_4sleeve_alpha_stack_daily.csv

Output:
  - logs/frontier/futures_data_gate/funding_regime_replay.csv
  - logs/frontier/futures_data_gate/funding_regime_result.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "logs" / "frontier" / "futures_data_gate"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_funding_panel() -> pd.DataFrame:
    p = ROOT / "data" / "frontier" / "funding" / "funding_panel_daily.parquet"
    df = pd.read_parquet(p)
    # Focus on majors — BTC + ETH avg funding
    keep = [c for c in ("btc_fund", "eth_fund") if c in df.columns]
    df["major_fund"] = df[keep].mean(axis=1)
    return df[["date", "major_fund"]].sort_values("date").reset_index(drop=True)


def build_regime(fund: pd.DataFrame) -> pd.DataFrame:
    out = fund.copy()
    # 30d trailing z-score of funding
    out["fund_mean30"] = out["major_fund"].rolling(30, min_periods=10).mean()
    out["fund_std30"] = out["major_fund"].rolling(30, min_periods=10).std()
    out["fund_z30"] = (out["major_fund"] - out["fund_mean30"]) / out["fund_std30"]
    # Regime classifier
    out["regime"] = "NORMAL"
    out.loc[out["fund_z30"] > 1.5, "regime"] = "OVER_LONG"       # de-risk (fade crowd)
    out.loc[out["fund_z30"] < -1.5, "regime"] = "OVER_SHORT"     # normal (no leverage cap)
    # Sizing map (NO leverage — cap at 1.0)
    MULT = {"OVER_LONG": 0.6, "NORMAL": 1.0, "OVER_SHORT": 1.0}
    out["multiplier"] = out["regime"].map(MULT).astype(float)
    return out


def replay(regime: pd.DataFrame) -> dict:
    bp = pd.read_csv(
        ROOT / "logs" / "portfolio_aggregator" / "recommended_4sleeve_alpha_stack_daily.csv",
        parse_dates=["date"],
    ).sort_values("date").set_index("date")
    bp["daily_ret_pct"] = bp["portfolio_equity"].pct_change().fillna(0.0) * 100.0
    reg = regime.set_index(pd.to_datetime(regime["date"])).drop(columns=["date"])
    df = bp.join(reg, how="left").ffill()
    df["multiplier"] = df["multiplier"].fillna(1.0)
    df["gated_ret_pct"] = df["daily_ret_pct"] * df["multiplier"]

    def metrics(r_pct: pd.Series, label: str) -> dict:
        r = r_pct.dropna() / 100.0
        mean = r.mean(); std = r.std()
        sh = (mean / std) * (365 ** 0.5) if std > 0 else 0.0
        down = r[r < 0]
        sr = (mean / down.std()) * (365 ** 0.5) if len(down) > 0 and down.std() > 0 else 0.0
        eq = (1.0 + r).cumprod()
        dd = float((eq / eq.cummax() - 1.0).min())
        n = len(r)
        cagr = (eq.iloc[-1]) ** (365.0 / n) - 1.0
        return {
            "label": label, "n_days": int(n), "cagr_pct": float(cagr * 100),
            "sharpe": float(sh), "sortino": float(sr), "max_dd_pct": float(dd * 100),
            "calmar": float((cagr / -dd) if dd < 0 else float("inf")),
        }

    res = {
        "flat": metrics(df["daily_ret_pct"], "flat"),
        "gated": metrics(df["gated_ret_pct"], "funding_gated"),
        "regime_day_counts": df["regime"].value_counts().to_dict(),
        "avg_multiplier": float(df["multiplier"].mean()),
    }
    df.reset_index().to_csv(OUT_DIR / "funding_regime_replay.csv", index=False)
    with open(OUT_DIR / "funding_regime_result.json", "w") as f:
        json.dump(res, f, indent=2, default=str)
    return res


def main() -> None:
    fund = load_funding_panel()
    print(f"[FUND] {len(fund)} days, {fund['date'].min().date()} -> {fund['date'].max().date()}")
    reg = build_regime(fund)
    print("\n[REGIME] distribution (all history):")
    print(reg["regime"].value_counts())

    res = replay(reg)
    print(f"\n[REPLAY] blend window day counts: {res['regime_day_counts']}")
    print(f"  avg multiplier: {res['avg_multiplier']:.4f}")
    print()
    print(f"  {'METRIC':<12} {'FLAT':>14} {'GATED':>14}")
    for k in ("n_days", "cagr_pct", "sharpe", "sortino", "max_dd_pct", "calmar"):
        flat_v = res["flat"].get(k, float("nan"))
        gated_v = res["gated"].get(k, float("nan"))
        fmt = ".4f" if k in ("sharpe", "sortino", "calmar") else ".2f"
        print(f"  {k:<12} {flat_v:>14{fmt}} {gated_v:>14{fmt}}")


if __name__ == "__main__":
    main()
