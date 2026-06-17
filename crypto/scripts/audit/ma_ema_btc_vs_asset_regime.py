"""ma_ema_btc_vs_asset_regime.py -- does asset's own regime matter vs BTC's?

User question (2026-05-20):
  "Check the BTC vs the asset's own. Does the asset's own allow us to disregard
   it from contention because it has weak signal, or does this not matter given
   we have setup in mind and not regime as our god?"

DESIGN:
  Take top-25% bear-performing assets (AR, SUI, FET, PEPE, LDO, ARKM, SUPER,
  CHZ, SOL, OP, NEAR, DYDX). For each:
    1. Compute asset's own 30d return at each event date (rolling)
    2. Bucket into asset_own_regime: bull (>+5%), chop, bear (<-5%), crash (<-15%)
    3. For asset's best MA pair (from earlier finding), slice the events both ways:
       - by btc_regime_30d
       - by asset_own_regime
    4. Compare: does asset_own_regime add discriminating signal vs btc_regime?
       Specifically, does the asset's best pair PERFORM DIFFERENTLY across
       asset_own_regime slices than across btc_regime slices?

Output: runs/audit/MA_EMA_PROFILE_2026_05_20/BTC_VS_ASSET_REGIME.md
"""
from __future__ import annotations

import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SNAP_PATH = ROOT / "runs" / "oracle_layer3" / "ma_ema_permutation" / "event_ma_snapshot.parquet"
CHIMERA_1D = ROOT / "data" / "processed" / "chimera" / "1d"
OUT = ROOT / "runs" / "audit" / "MA_EMA_PROFILE_2026_05_20" / "BTC_VS_ASSET_REGIME.md"

BUCKET_COST_FRAC_TAKER = {"BLUE": 0.0018, "STEADY": 0.0012, "VOLATILE": 0.0013, "DEGEN": 0.0010}

# Top-25% bear performers and their best LO pair (from previous finding)
TARGETS = [
    ("AR",    "VOLATILE", "SMA", 13, 36),
    ("SUI",   "VOLATILE", "SMA", 42, 43),
    ("FET",   "DEGEN",    "SMA", 13, 23),
    ("PEPE",  "VOLATILE", "SMA", 25, 29),
    ("LDO",   "VOLATILE", "SMA", 94, 100),
    ("ARKM",  "DEGEN",    "SMA", 3, 6),
    ("SUPER", "DEGEN",    "SMA", 25, 27),
    ("CHZ",   "VOLATILE", "SMA", 19, 36),
    ("SOL",   "STEADY",   "SMA", 15, 19),
    ("OP",    "VOLATILE", "SMA", 24, 26),
    ("NEAR",  "VOLATILE", "SMA", 17, 19),
    ("DYDX",  "VOLATILE", "SMA", 23, 45),
    # And ETH for contrast (loser in bear)
    ("ETH",   "BLUE",     "SMA", 9, 12),
]


def asset_own_regime(ret_30d: float) -> str:
    """Same buckets as btc_regime_30d in the project."""
    if pd.isna(ret_30d): return "unknown"
    if ret_30d <= -0.15: return "crash"
    if ret_30d <= -0.05: return "bear"
    if ret_30d >= 0.05:  return "bull"
    return "chop"


def load_asset_close_series(asset: str) -> pd.DataFrame:
    """Load chimera 1d for one asset, return (date, close, ret_30d)."""
    sym = asset.lower() + "usdt"
    files = sorted(glob.glob(str(CHIMERA_1D / f"{sym}_v51_chimera_1d_*.parquet")))
    if not files: return None
    df = pl.read_parquet(files[-1], columns=["timestamp", "close"]).to_pandas()
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)
    df["ret_30d"] = df["close"].pct_change(30)
    df["asset_own_regime"] = df["ret_30d"].apply(asset_own_regime)
    return df[["date", "close", "ret_30d", "asset_own_regime"]]


def eval_pair_long_only(events: pd.DataFrame, ma_type: str, fast: int, slow: int, cost: float) -> dict:
    """Long-only PnL on a slice. Returns aggregate stats."""
    col_f = f"{ma_type}_{fast}"; col_s = f"{ma_type}_{slow}"
    if col_f not in events.columns or col_s not in events.columns:
        return None
    sub = events[events[col_f].notna() & events[col_s].notna()].copy()
    sub["sig"] = np.sign(sub[col_f] - sub[col_s])
    sub["mag"] = sub["magnitude_signed"] / 100.0
    fired = sub[sub["sig"] == 1]
    if len(fired) < 3:
        return {"n": len(fired), "mean_pct": None, "hit": None, "sum_pct": None, "sharpe": None}
    pnl = fired["mag"] - cost
    return {
        "n": len(fired),
        "mean_pct": float(pnl.mean() * 100),
        "hit": float((pnl > 0).mean()),
        "sum_pct": float(pnl.sum() * 100),
        "sharpe": float(pnl.mean() / pnl.std()) if pnl.std() > 1e-9 else 0.0,
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("Loading event snapshot...")
    snap = pl.read_parquet(SNAP_PATH).to_pandas()
    snap["date"] = pd.to_datetime(snap["date"])
    print(f"  events: {len(snap):,} across {snap['asset'].nunique()} assets")

    rows_all = []
    sliced_all = {}
    for asset, bucket, ma_type, fast, slow in TARGETS:
        print(f"\n=== {asset} ({bucket}) — {ma_type}({fast}, {slow}) ===")
        ev = snap[snap["asset"] == asset].copy()
        if len(ev) == 0:
            print(f"  no events; skip")
            continue
        # Load asset's own close series, merge ret_30d → asset_own_regime
        own = load_asset_close_series(asset)
        if own is None:
            print(f"  no chimera; skip")
            continue
        ev = ev.merge(own[["date", "ret_30d", "asset_own_regime"]], on="date", how="left")
        cost = BUCKET_COST_FRAC_TAKER[bucket]

        # Overall
        full = eval_pair_long_only(ev, ma_type, fast, slow, cost)
        print(f"  Overall: n={full['n']}, mean={full['mean_pct']:+.3f}%, hit={full['hit']*100:.1f}%, sum={full['sum_pct']:+.2f}%, Sh={full['sharpe']:+.4f}" if full and full['mean_pct'] is not None else "  Overall: insufficient events")

        # By BTC regime
        print(f"  By BTC regime:")
        btc_slices = {}
        for reg in ("bull", "chop", "bear", "crash"):
            sub = ev[ev["btc_regime_30d"] == reg]
            stats = eval_pair_long_only(sub, ma_type, fast, slow, cost)
            btc_slices[reg] = stats
            if stats and stats["mean_pct"] is not None:
                print(f"    btc_{reg:<6} n={stats['n']:3d} mean={stats['mean_pct']:+.3f}% hit={stats['hit']*100:5.1f}% sum={stats['sum_pct']:+.2f}% Sh={stats['sharpe']:+.4f}")
            else:
                print(f"    btc_{reg:<6} insufficient ({stats['n'] if stats else 0} fires)")

        # By asset's own regime
        print(f"  By asset's own regime:")
        own_slices = {}
        for reg in ("bull", "chop", "bear", "crash"):
            sub = ev[ev["asset_own_regime"] == reg]
            stats = eval_pair_long_only(sub, ma_type, fast, slow, cost)
            own_slices[reg] = stats
            if stats and stats["mean_pct"] is not None:
                print(f"    own_{reg:<6} n={stats['n']:3d} mean={stats['mean_pct']:+.3f}% hit={stats['hit']*100:5.1f}% sum={stats['sum_pct']:+.2f}% Sh={stats['sharpe']:+.4f}")
            else:
                print(f"    own_{reg:<6} insufficient ({stats['n'] if stats else 0} fires)")

        sliced_all[asset] = {"overall": full, "btc": btc_slices, "own": own_slices,
                              "bucket": bucket, "pair": f"{ma_type}({fast}, {slow})"}

    # WRITE REPORT
    lines = ["# BTC vs Asset's Own Regime — empirical test (2026-05-20)\n"]
    lines.append("**Question**: when filtering by regime, does the ASSET'S OWN 30d return")
    lines.append("regime add discriminating signal versus the BTC 30d return regime?")
    lines.append("Or are they so correlated that BTC works as a proxy?\n")
    lines.append("**Method**: for each top-25%-bear-performing asset, take its best LO MA/EMA")
    lines.append("pair and slice its events by (a) BTC regime, (b) asset's own 30d return.")
    lines.append("Asset regime cuts: bull≥+5%, chop in (-5%,+5%), bear in (-15%,-5%], crash≤-15%.\n")
    lines.append("**Important**: same pair, same data, just two different slicings of the regime.\n")

    # Per-asset detailed table
    for asset, info in sliced_all.items():
        lines.append(f"## {asset} — {info['pair']} ({info['bucket']})\n")
        o = info["overall"]
        if o and o["mean_pct"] is not None:
            lines.append(f"**Overall**: n={o['n']}, mean={o['mean_pct']:+.3f}%, hit={o['hit']*100:.1f}%, sum={o['sum_pct']:+.2f}%, Sh={o['sharpe']:+.4f}\n")

        lines.append("### By BTC regime\n")
        lines.append("| regime | n | mean % | hit % | sum % | Sharpe |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for reg in ("bull", "chop", "bear", "crash"):
            s = info["btc"].get(reg)
            if s and s["mean_pct"] is not None:
                lines.append(f"| {reg} | {s['n']} | {s['mean_pct']:+.3f} | {s['hit']*100:.1f} | {s['sum_pct']:+.2f} | {s['sharpe']:+.4f} |")
            else:
                lines.append(f"| {reg} | {s['n'] if s else 0} | — | — | — | — |")

        lines.append("\n### By asset's own regime\n")
        lines.append("| regime | n | mean % | hit % | sum % | Sharpe |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for reg in ("bull", "chop", "bear", "crash"):
            s = info["own"].get(reg)
            if s and s["mean_pct"] is not None:
                lines.append(f"| {reg} | {s['n']} | {s['mean_pct']:+.3f} | {s['hit']*100:.1f} | {s['sum_pct']:+.2f} | {s['sharpe']:+.4f} |")
            else:
                lines.append(f"| {reg} | {s['n'] if s else 0} | — | — | — | — |")
        lines.append("")

    # Cross-asset summary: where does asset_own DIFFER from btc regime?
    lines.append("## SUMMARY — does asset-own regime add signal vs BTC regime?\n")
    lines.append("For each asset, we compare the Sharpe-proxy distribution across the 4 regimes")
    lines.append("when sliced by BTC vs sliced by asset's own. If the asset-own slicing gives a")
    lines.append("WIDER range of Sharpes (better discrimination), it's the more useful regime axis.\n")
    lines.append("| asset | btc-slice Sharpe range | own-slice Sharpe range | own wider? |")
    lines.append("|---|---|---|---:|")
    n_own_wider = 0
    n_total = 0
    for asset, info in sliced_all.items():
        btc_sharpes = [s["sharpe"] for s in info["btc"].values() if s and s["mean_pct"] is not None]
        own_sharpes = [s["sharpe"] for s in info["own"].values() if s and s["mean_pct"] is not None]
        if not btc_sharpes or not own_sharpes:
            lines.append(f"| {asset} | insufficient data | insufficient data | — |")
            continue
        btc_range = max(btc_sharpes) - min(btc_sharpes)
        own_range = max(own_sharpes) - min(own_sharpes)
        own_wider = own_range > btc_range
        n_total += 1
        if own_wider: n_own_wider += 1
        lines.append(f"| {asset} | {min(btc_sharpes):+.3f} to {max(btc_sharpes):+.3f} (Δ={btc_range:.3f}) | "
                     f"{min(own_sharpes):+.3f} to {max(own_sharpes):+.3f} (Δ={own_range:.3f}) | "
                     f"{'YES' if own_wider else 'no'} |")
    lines.append(f"\n**Aggregate: own-regime gives WIDER Sharpe spread on {n_own_wider}/{n_total} assets.**\n")
    if n_total > 0:
        if n_own_wider / n_total > 0.6:
            lines.append("**Verdict**: asset's own regime is MORE discriminating than BTC regime.")
            lines.append("Use asset's own 30d regime as the primary gate; BTC as auxiliary context.\n")
        elif n_own_wider / n_total < 0.4:
            lines.append("**Verdict**: BTC regime is more (or equally) discriminating. Keep BTC")
            lines.append("regime as the gate; asset's own adds little.\n")
        else:
            lines.append("**Verdict**: MIXED. Some assets respond to BTC regime more, others to own.")
            lines.append("Implication: use BOTH as features in a per-asset ranker, not a single gate.\n")

    # Check: does asset-own-bear KILL performance even when btc-bear has done OK?
    lines.append("## The setup-vs-regime question — does the asset's best pair still WORK in asset-own-bear?\n")
    lines.append("If asset's best pair is profitable in asset's own bear, then asset-own-regime")
    lines.append("DOES NOT need to be a strict gate (the setup itself works regardless).")
    lines.append("If asset's best pair fails in asset's own bear, then asset-own-regime IS a useful filter.\n")
    lines.append("| asset | own-bear Sharpe | own-bull Sharpe | own-chop Sharpe | does setup survive own-bear? |")
    lines.append("|---|---:|---:|---:|---:|")
    survive_own_bear = 0
    n_with_own_bear = 0
    for asset, info in sliced_all.items():
        ob = info["own"].get("bear")
        ou = info["own"].get("bull")
        oc = info["own"].get("chop")
        if not ob or ob["mean_pct"] is None:
            lines.append(f"| {asset} | (n={ob['n'] if ob else 0}, insufficient) | "
                         f"{ou['sharpe']:+.3f}" if ou and ou['mean_pct'] is not None else "—"
                         f" | "
                         f"{oc['sharpe']:+.3f}" if oc and oc['mean_pct'] is not None else "—"
                         f" | — |")
            continue
        n_with_own_bear += 1
        survives = ob["sharpe"] > 0
        if survives: survive_own_bear += 1
        ou_s = f"{ou['sharpe']:+.3f}" if ou and ou["mean_pct"] is not None else "—"
        oc_s = f"{oc['sharpe']:+.3f}" if oc and oc["mean_pct"] is not None else "—"
        lines.append(f"| {asset} | {ob['sharpe']:+.3f} (n={ob['n']}) | {ou_s} | {oc_s} | "
                     f"{'YES (survives)' if survives else 'NO (gate needed)'} |")

    lines.append(f"\n**Setup survives asset-own-bear in {survive_own_bear}/{n_with_own_bear} assets**.\n")
    if n_with_own_bear > 0:
        survive_pct = survive_own_bear / n_with_own_bear * 100
        if survive_pct >= 70:
            lines.append("**Implication**: most setups survive even asset-own-bear — under setup-first doctrine,")
            lines.append("regime gates are MILDLY USEFUL but not god. Setup IS the signal.\n")
        elif survive_pct <= 30:
            lines.append("**Implication**: most setups break in asset-own-bear — regime IS a useful gate.")
            lines.append("Don't deploy when asset's own 30d is in bear. The setup needs trend tailwind.\n")
        else:
            lines.append("**Implication**: per-asset variance — some setups need regime, others don't.")
            lines.append("Per-asset profile must tag which regimes each cell qualifies in (asset-conditional).\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
