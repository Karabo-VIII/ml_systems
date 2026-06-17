"""Alpha turn-007: per-sleeve daily returns panel for R2 rescue.

Emits logs/portfolio_aggregator/recommended_4sleeve_per_sleeve_returns.csv
with one column per sleeve (daily return %) + a date index.

This lets Bravo test per-sleeve conditional sizing (R2 rescue) without
re-running the aggregator. Same as what the aggregator internally uses
but with per-sleeve returns kept separate instead of blended.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEEDS_DIR = ROOT / "logs" / "paper_trader_v2" / "seeds"
OUT = ROOT / "logs" / "portfolio_aggregator" / "recommended_4sleeve_per_sleeve_returns.csv"

SLEEVES = {
    "xsec_K10_10_FULL_dneut_U50": "pt_xsec_K10_10_FULL_dneut_U50",
    "frontier_dib_flow_both":     "pt_frontier_dib_flow_both",
    "asym_breakout":              "pt_asym_breakout",
    "asym_vol_expansion":         "pt_asym_vol_expansion",
}


def load_sleeve(seed: str) -> pd.DataFrame:
    path = SEEDS_DIR / seed / "daily_snapshot.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[["date", "total_equity"]].copy()
    # total_equity is the daily mark -- derive daily return
    df["daily_ret_pct"] = df["total_equity"].pct_change().fillna(0) * 100.0
    return df[["date", "daily_ret_pct"]].rename(columns={"daily_ret_pct": "ret"})


def main() -> None:
    panels = []
    for name, seed in SLEEVES.items():
        s = load_sleeve(seed).rename(columns={"ret": name})
        panels.append(s.set_index("date"))
    wide = pd.concat(panels, axis=1).sort_index()
    # Equal-weighted blend-check (sanity column) — should match blend CSV ret
    wide["blend_EW"] = wide[list(SLEEVES.keys())].mean(axis=1)
    wide.to_csv(OUT)
    print(f"[OK] wrote {OUT}")
    print(f"shape={wide.shape}  date range: {wide.index.min().date()} -> {wide.index.max().date()}")
    # Quick per-sleeve stats
    for name in SLEEVES:
        s = wide[name].dropna()
        sh = (s.mean() / s.std()) * (365 ** 0.5) if s.std() > 0 else 0.0
        print(f"  {name:32s} n={len(s):4d} mean={s.mean():+.3f}%  std={s.std():.3f}%  Sh_365={sh:+.2f}")
    print()
    blend = wide["blend_EW"].dropna()
    sh = (blend.mean() / blend.std()) * (365 ** 0.5)
    print(f"  {'blend_EW (sanity check)':32s} n={len(blend):4d} mean={blend.mean():+.3f}%  std={blend.std():.3f}%  Sh_365={sh:+.2f}")


if __name__ == "__main__":
    main()
