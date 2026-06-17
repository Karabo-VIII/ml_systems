"""ma_ema_per_asset_dna_cards.py -- generate per-asset DNA visualization cards.

For each asset with profile cells, output a card showing:
  - Bucket + sector
  - Top-3 cells (best overall by Sharpe)
  - Cousin set (deploy-grade complementary triple)
  - Regime-tag distribution
  - Per-regime stats for the top cell
  - OOS performance breakdown (if available)
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "data" / "processed" / "per_asset_ma_ema_profile.parquet"
OOS_TRADES = ROOT / "runs" / "audit" / "MA_EMA_PROFILE_2026_05_20" / "oos_per_asset_trades_best.csv"
OOS_BREAKDOWN = ROOT / "runs" / "audit" / "MA_EMA_PROFILE_2026_05_20" / "oos_per_asset_breakdown.csv"
OUT = ROOT / "runs" / "audit" / "MA_EMA_PROFILE_2026_05_20" / "PER_ASSET_DNA_CARDS.md"


def parse_regime(j: str) -> dict:
    """Parse the JSON-encoded per-regime stats."""
    try:
        d = json.loads(j)
        return d if d else {"n": 0}
    except Exception:
        return {"n": 0}


def render_card(asset: str, asset_cells: pd.DataFrame, oos_row: pd.Series = None) -> list:
    lines = []
    bucket = asset_cells["bucket"].iloc[0]
    n_cells = len(asset_cells)
    n_cousin = int(asset_cells["is_cousin_set_member"].sum())

    lines.append(f"### {asset} ({bucket})")
    lines.append("")
    lines.append(f"**Profile**: {n_cells} qualifying cells / **{n_cousin} cousin-set members** (deploy-grade)")
    if oos_row is not None and not oos_row.empty:
        coverage = oos_row.get("coverage_pct")
        capture = oos_row.get("capture_pct")
        coverage_s = f"{coverage:.1f}%" if pd.notna(coverage) else "—"
        capture_s = f"{capture:+.1f}%" if pd.notna(capture) else "—"
        lines.append(f"**OOS perf** (2024-05-16 → 2025-03-15, best exit `tight_trail_5_3_14d`):")
        lines.append(f"  - Trades: **{int(oos_row['n_trades'])}** | Win rate: **{oos_row['win_rate']:.1f}%** | "
                     f"Mean ret/trade: **{oos_row['mean_ret_pct']:+.3f}%** | Sum ret: **{oos_row['sum_ret_pct']:+.2f}%**")
        lines.append(f"  - Oracle long avail: {oos_row['oracle_long_avail_pct']:+.2f}% across {int(oos_row['oracle_pos_days'])} +1%-days")
        lines.append(f"  - **Coverage**: {coverage_s} (fraction of asset's positive-event days that we entered)")
        lines.append(f"  - **Capture**: {capture_s} (realized PnL / oracle long availability)")
        lines.append(f"  - **NAV contribution**: {oos_row['contribution_to_nav_pct']:+.4f}% (this asset's slice of total portfolio growth)")
    lines.append("")

    # Cousin set (deploy-grade)
    cousins = asset_cells[asset_cells["is_cousin_set_member"]].sort_values("sharpe", ascending=False)
    if len(cousins):
        lines.append("**Cousin set (deploy-grade, complementary cells)**:")
        lines.append("")
        lines.append("| type | (fast, slow) | n_VAL | mean % | hit % | Sharpe | regime tag |")
        lines.append("|---|---|---:|---:|---:|---:|---|")
        for _, c in cousins.iterrows():
            lines.append(f"| {c['ma_type']} | ({c['fast']}, {c['slow']}) | {c['overall_n']} | "
                         f"{c['overall_mean_pct']:+.3f} | {c['overall_hit']*100:.1f} | "
                         f"{c['sharpe']:+.4f} | `{c['regime_tag']}` |")
        lines.append("")

    # Regime tag distribution
    tag_counts = asset_cells["regime_tag"].value_counts()
    lines.append(f"**Regime-tag distribution** (across {n_cells} cells): " +
                 ", ".join(f"`{t}`={n}" for t, n in tag_counts.items()))
    lines.append("")

    # Top-1 cell regime breakdown (own_regime)
    top = asset_cells.sort_values("sharpe", ascending=False).iloc[0]
    own = {r: parse_regime(top[f"own_{r}"]) for r in ("bull", "chop", "bear", "crash")}
    btc = {r: parse_regime(top[f"btc_{r}"]) for r in ("bull", "chop", "bear", "crash")}
    lines.append(f"**Top cell** ({top['ma_type']}({top['fast']}, {top['slow']})) — per-regime stats:")
    lines.append("")
    lines.append("| regime | own_n | own_mean % | own_hit % | own_Sharpe | btc_n | btc_mean % | btc_hit % | btc_Sharpe |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in ("bull", "chop", "bear", "crash"):
        o = own[r]; b = btc[r]
        ow_n = o.get("n", 0); ow_m = o.get("mean", 0); ow_h = o.get("hit", 0); ow_s = o.get("sharpe", 0)
        bt_n = b.get("n", 0); bt_m = b.get("mean", 0); bt_h = b.get("hit", 0); bt_s = b.get("sharpe", 0)
        lines.append(f"| {r} | {ow_n} | {ow_m:+.3f} | {ow_h*100:.1f} | {ow_s:+.4f} | "
                     f"{bt_n} | {bt_m:+.3f} | {bt_h*100:.1f} | {bt_s:+.4f} |")
    lines.append("")
    return lines


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    profile = pd.read_parquet(PROFILE)
    print(f"Profile: {len(profile)} cells / {profile['asset'].nunique()} assets")

    oos_breakdown = None
    if OOS_BREAKDOWN.exists():
        oos_breakdown = pd.read_csv(OOS_BREAKDOWN)
        print(f"OOS breakdown: {len(oos_breakdown)} assets")

    lines = [
        "# MA/EMA Per-Asset DNA Cards (2026-05-20)",
        "",
        f"**Source**: per_asset_ma_ema_profile.parquet + OOS breakdown",
        f"**OOS window**: 2024-05-16 → 2025-03-15 (~303 days), best exit `tight_trail_5_3_14d`",
        "",
        "Per-asset DNA = (cousin set, regime tags, per-regime stats, OOS performance, coverage, capture)",
        "",
        "## Asset cards (sorted by OOS NAV contribution)",
        "",
    ]

    # Sort assets by OOS NAV contribution if available, else by best Sharpe
    if oos_breakdown is not None:
        ordered_assets = oos_breakdown.sort_values("contribution_to_nav_pct", ascending=False)["asset"].tolist()
        # Append assets not in OOS at the end
        in_oos = set(ordered_assets)
        rest = [a for a in profile["asset"].unique() if a not in in_oos]
        ordered_assets += sorted(rest)
    else:
        ordered_assets = sorted(profile["asset"].unique())

    for asset in ordered_assets:
        asset_cells = profile[profile["asset"] == asset]
        if asset_cells.empty: continue
        oos_row = None
        if oos_breakdown is not None:
            matching = oos_breakdown[oos_breakdown["asset"] == asset]
            if not matching.empty:
                oos_row = matching.iloc[0]
        card = render_card(asset, asset_cells, oos_row)
        lines.extend(card)
        lines.append("---")
        lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] wrote {OUT}")
    print(f"  cards: {len(ordered_assets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
