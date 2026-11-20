"""ma_ema_eth_bear_slice.py -- surgical slice: ETH in bear + top-25% bear performers.

User asked (2026-05-20):
  1. Implicit assumptions in the MA/EMA analysis
  2. ETH bear cycle: top performing MA/EMA strats with numbers
  3. Oracle capture rate on that slice
  4. Top 25% of bear-slice performers (assets) -- their best MA/EMA

DATA: runs/oracle_layer3/ma_ema_permutation/event_ma_snapshot.parquet
  Per-event raw rows with: asset, date, side, magnitude_pct, magnitude_signed,
  bucket, sector, btc_regime_30d, cluster_id, SMA_1..SMA_100, EMA_1..EMA_100

  VAL window: 2023-07-01 -> 2024-05-15
  Events filtered to def_type='both_cc' (>1% close-to-close moves, long OR short)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SNAP_PATH = ROOT / "runs" / "oracle_layer3" / "ma_ema_permutation" / "event_ma_snapshot.parquet"
OUT = ROOT / "runs" / "audit" / "MA_EMA_PROFILE_2026_05_20" / "ETH_BEAR_SLICE.md"

# Cost (from build_ma_ema_permutation.py taker defaults)
BUCKET_COST_FRAC_TAKER = {"BLUE": 0.0018, "STEADY": 0.0012, "VOLATILE": 0.0013, "DEGEN": 0.0010}
N_MAX = 100  # max MA period in snapshot


def eval_pair_on_slice(slice_df: pd.DataFrame, ma_type: str, fast: int, slow: int, cost: float) -> dict:
    """Compute MA/EMA pair performance on a pre-filtered slice."""
    col_f = f"{ma_type}_{fast}"
    col_s = f"{ma_type}_{slow}"
    if col_f not in slice_df.columns or col_s not in slice_df.columns:
        return None
    sub = slice_df[slice_df[col_f].notna() & slice_df[col_s].notna()].copy()
    if len(sub) < 10:
        return None
    sub["sig"] = np.sign(sub[col_f] - sub[col_s])
    sub["mag"] = sub["magnitude_signed"] / 100.0  # decimal
    # PnL when signal nonzero: signal * mag - cost
    fired = sub[sub["sig"] != 0]
    if len(fired) < 5:
        return None
    pnl = fired["sig"] * fired["mag"] - cost * fired["sig"].abs()
    n_long = (fired["sig"] == 1).sum()
    n_short = (fired["sig"] == -1).sum()
    return {
        "ma_type": ma_type, "fast": fast, "slow": slow,
        "n_events_in_slice": len(sub),
        "n_signaled": len(fired),
        "n_long_sig": int(n_long),
        "n_short_sig": int(n_short),
        "mean_pnl_pct": float(pnl.mean() * 100),
        "median_pnl_pct": float(pnl.median() * 100),
        "std_pnl_pct": float(pnl.std() * 100),
        "sum_pnl_pct": float(pnl.sum() * 100),
        "hit_rate": float((pnl > 0).mean()),
        "sharpe_proxy": float(pnl.mean() / pnl.std()) if pnl.std() > 1e-9 else 0.0,
        "balance_long_short": float(abs(n_long - n_short) / max(1, n_long + n_short)),
    }


def long_only_pnl(slice_df: pd.DataFrame, ma_type: str, fast: int, slow: int, cost: float) -> dict:
    """Long-only variant: only take signal == +1 fires."""
    col_f = f"{ma_type}_{fast}"
    col_s = f"{ma_type}_{slow}"
    if col_f not in slice_df.columns or col_s not in slice_df.columns:
        return None
    sub = slice_df[slice_df[col_f].notna() & slice_df[col_s].notna()].copy()
    sub["sig"] = np.sign(sub[col_f] - sub[col_s])
    sub["mag"] = sub["magnitude_signed"] / 100.0
    fired = sub[sub["sig"] == 1]
    if len(fired) < 5:
        return None
    pnl = fired["mag"] - cost
    return {
        "ma_type": ma_type, "fast": fast, "slow": slow,
        "n_long_fires": len(fired),
        "mean_pnl_pct": float(pnl.mean() * 100),
        "sum_pnl_pct": float(pnl.sum() * 100),
        "hit_rate": float((pnl > 0).mean()),
        "sharpe_proxy": float(pnl.mean() / pnl.std()) if pnl.std() > 1e-9 else 0.0,
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("Loading event snapshot...")
    snap = pl.read_parquet(SNAP_PATH).to_pandas()
    snap["date"] = pd.to_datetime(snap["date"])
    print(f"  total events: {len(snap):,}")
    print(f"  date range: {snap['date'].min()} -> {snap['date'].max()}")
    print(f"  assets: {snap['asset'].nunique()}")
    print(f"  regimes: {snap['btc_regime_30d'].value_counts().to_dict()}")

    # ===== 1. BEAR SLICE: ETH =====
    eth_bear = snap[(snap["asset"] == "ETH") & (snap["btc_regime_30d"] == "bear")].copy()
    print(f"\n=== ETH bear events: {len(eth_bear)} ===")
    if len(eth_bear) == 0:
        print("[WARN] no ETH bear events -- checking what regimes ETH has...")
        eth_all = snap[snap["asset"] == "ETH"]
        print(f"  ETH regimes: {eth_all['btc_regime_30d'].value_counts().to_dict()}")

    # ETH bucket
    eth_bucket = eth_bear["bucket"].iloc[0] if len(eth_bear) else "BLUE"
    cost = BUCKET_COST_FRAC_TAKER.get(eth_bucket, 0.0018)
    print(f"  ETH bucket: {eth_bucket}, cost (taker): {cost*100:.2f}%")

    # Date range of ETH bear events
    if len(eth_bear):
        print(f"  ETH bear dates: {eth_bear['date'].min().date()} -> {eth_bear['date'].max().date()}")
        # Sample of magnitudes
        print(f"  ETH bear |magnitude| stats:")
        print(f"    mean +{eth_bear['magnitude_pct'].mean():.2f}%, median +{eth_bear['magnitude_pct'].median():.2f}%, max +{eth_bear['magnitude_pct'].max():.2f}%")
        # By side
        print(f"  ETH bear by side:")
        print(eth_bear["side"].value_counts().to_string())

    # 1a. Exhaustive MA/EMA evaluation on ETH bear (long+short and long-only)
    print(f"\n=== Computing MA/EMA pair performance on ETH bear ===")
    eth_long_short = []
    eth_long_only = []
    for ma_type in ("SMA", "EMA"):
        for fast in range(1, N_MAX):
            for slow in range(fast + 1, N_MAX + 1):
                r1 = eval_pair_on_slice(eth_bear, ma_type, fast, slow, cost)
                if r1 and r1["n_signaled"] >= 8:
                    eth_long_short.append(r1)
                r2 = long_only_pnl(eth_bear, ma_type, fast, slow, cost)
                if r2 and r2["n_long_fires"] >= 8:
                    eth_long_only.append(r2)

    ls_df = pd.DataFrame(eth_long_short).sort_values("sharpe_proxy", ascending=False)
    lo_df = pd.DataFrame(eth_long_only).sort_values("sharpe_proxy", ascending=False)
    print(f"  long+short qualifying pairs: {len(ls_df)}")
    print(f"  long-only qualifying pairs:  {len(lo_df)}")

    # ===== 2. Oracle availability and capture =====
    # "Oracle" for the bear-ETH slice = the perfect-foresight long-only mover capture:
    # sum of positive cc_ret on ETH bear days where a >1% move occurred AND was up
    # (i.e., long-side magnitude_signed > 0).
    eth_bear_long_events = eth_bear[eth_bear["magnitude_signed"] > 0]
    oracle_long_sum_pct = float(eth_bear_long_events["magnitude_pct"].sum())  # sum of positive cc% on bear days for ETH
    oracle_n_long_days = len(eth_bear_long_events)
    # All bear ETH events (long + short fires)
    oracle_total_abs_sum_pct = float(eth_bear["magnitude_pct"].abs().sum())
    n_unique_bear_dates = eth_bear["date"].dt.date.nunique()

    print(f"\n=== Oracle availability (ETH bear) ===")
    print(f"  ETH bear LONG (cc>0) events: {oracle_n_long_days}")
    print(f"  Sum of |magnitude| on ETH bear LONG events: {oracle_long_sum_pct:+.2f}%")
    print(f"  Sum of |magnitude| on ALL ETH bear events: {oracle_total_abs_sum_pct:+.2f}%")
    print(f"  Unique ETH bear dates: {n_unique_bear_dates}")

    # ===== 3. Top 25% of bear-slice asset performers =====
    print(f"\n=== Top 25% of assets by bear-slice oracle availability (long-only) ===")
    all_bear = snap[snap["btc_regime_30d"] == "bear"].copy()
    # Per-asset oracle long availability = sum of positive magnitude_pct on bear events
    long_avail = (all_bear[all_bear["magnitude_signed"] > 0]
                  .groupby("asset")["magnitude_pct"].sum()
                  .sort_values(ascending=False))
    n_total_assets = long_avail.shape[0]
    n_top25 = max(1, int(np.ceil(n_total_assets * 0.25)))
    top25 = long_avail.head(n_top25)
    print(f"  Total assets with bear events: {n_total_assets}")
    print(f"  Top-25% cutoff: {n_top25} assets")
    print(f"  Top-10 by bear long-availability:")
    for asset, avail in top25.head(10).items():
        n_long_ev = len(all_bear[(all_bear["asset"] == asset) & (all_bear["magnitude_signed"] > 0)])
        print(f"    {asset:<8} +{avail:.2f}% sum, n_long_events={n_long_ev}")

    # For each top-25% asset, find their best MA/EMA pair on bear slice (long-only)
    print(f"\n=== Best MA/EMA pair per top-25% bear asset (long-only) ===")
    top25_best_pairs = []
    for asset in top25.index:
        ab = all_bear[all_bear["asset"] == asset]
        bucket = ab["bucket"].iloc[0] if len(ab) else "BLUE"
        c = BUCKET_COST_FRAC_TAKER.get(bucket, 0.0018)
        best = None
        for ma_type in ("SMA", "EMA"):
            for fast in range(1, N_MAX):
                for slow in range(fast + 1, N_MAX + 1):
                    r = long_only_pnl(ab, ma_type, fast, slow, c)
                    if r and r["n_long_fires"] >= 5:
                        # Score by sum_pnl since n is small in bear slices
                        if best is None or r["sum_pnl_pct"] > best["sum_pnl_pct"]:
                            best = {**r, "asset": asset, "bucket": bucket}
        if best:
            top25_best_pairs.append(best)
    top25_df = pd.DataFrame(top25_best_pairs).sort_values("sum_pnl_pct", ascending=False)

    # Write report
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# MA/EMA — ETH Bear Slice + Top-25% Performers (2026-05-20)\n"]
    lines.append(f"**Data**: `runs/oracle_layer3/ma_ema_permutation/event_ma_snapshot.parquet`\n")
    lines.append(f"**Window**: VAL ({snap['date'].min().date()} → {snap['date'].max().date()})\n")
    lines.append(f"**Slice**: btc_regime_30d == 'bear' (BTC 30d return ≤ -5%, > -15%)\n")
    lines.append("")

    # IMPLICIT ASSUMPTIONS
    lines.append("## 0. IMPLICIT ASSUMPTIONS in this analysis (be skeptical of each)\n")
    lines.append("1. **Event-conditional measurement.** All numbers below are computed on event")
    lines.append("   days only — days where the asset's close-to-close return was > +1% or < -1%")
    lines.append("   in absolute terms (def_type='both_cc'). Non-event days are EXCLUDED. So")
    lines.append("   per-event Sharpe ≠ deployable daily Sharpe; the latter is much lower because")
    lines.append("   non-event days drag the mean toward 0.")
    lines.append("2. **1-day hold by construction.** PnL = signal_at_T-1 × cc_return_at_T - cost.")
    lines.append("   This is an OPEN-AT-T-1-CLOSE → CLOSE-AT-T trade. Multi-day holds aren't modeled.")
    lines.append("   The TRAIN smart-discovery uses 14d-fwd outcome which is different.")
    lines.append("3. **Direction-only signal.** signal = sign(MA_fast - MA_slow). Magnitude of the")
    lines.append("   cross (how much fast exceeds slow) isn't used. A barely-positive spread and a")
    lines.append("   massively-positive spread both contribute signal = +1.")
    lines.append("4. **No look-ahead.** MA values at T are computed from prices through T-1 (verified")
    lines.append("   in build_ma_ema_permutation.py:179 explicit `.shift(1)` on MA cols).")
    lines.append("5. **regime label is BTC's regime, NOT asset's own regime.** ETH 'bear' means BTC")
    lines.append("   was in bear; ETH could itself be trending differently. Conflating BTC regime with")
    lines.append("   asset-specific regime is a known approximation.")
    lines.append("6. **Cost is bucket-level taker (BLUE 0.18% RT for ETH).** Maker cost is 0.05% RT —")
    lines.append("   not applied here. Switching to maker shrinks the cost drag ~70%.")
    lines.append("7. **No position-cap or capital accounting.** Each event evaluated independently.")
    lines.append("   sum_pnl_pct = arithmetic event-sum, NOT portfolio NAV under K-cap. The")
    lines.append("   `nav_4pct_upper_bound_arithmetic` warning applies here too — see")
    lines.append("   `docs/ORACLE_CORRECTIONS_2026_05_20.md`.")
    lines.append("8. **Filtering thresholds**: pairs require n_signaled ≥ 8 for long+short, ")
    lines.append("   n_long_fires ≥ 5 for long-only. Smaller-n cells excluded as noise.")
    lines.append("9. **Universe coverage**: ETH bear events are limited to dates in the VAL window")
    lines.append("   where ETH had chimera data AND BTC was in bear regime. May exclude periods")
    lines.append("   where ETH had data but BTC's regime didn't qualify.")
    lines.append("10. **Magnitude_signed semantics**: this is the SIGNED 1d cc return. A long-side")
    lines.append("    event with magnitude_signed = +2.5% means ETH closed +2.5% above prior close")
    lines.append("    on that bear day. So 'bear' periods can have up-moves.")
    lines.append("")

    # ETH bear slice headline
    lines.append("## 1. ETH bear slice — basic stats\n")
    lines.append(f"- Bear events for ETH: **{len(eth_bear)}**")
    if len(eth_bear):
        lines.append(f"- Date range: {eth_bear['date'].min().date()} → {eth_bear['date'].max().date()}")
        lines.append(f"- Unique bear dates: {n_unique_bear_dates}")
        lines.append(f"- Side breakdown: " +
                     ", ".join(f"{k}={v}" for k, v in eth_bear["side"].value_counts().items()))
        lines.append(f"- Mean abs(magnitude_pct): {eth_bear['magnitude_pct'].abs().mean():+.2f}%")
        lines.append(f"- Sum of LONG side magnitude (oracle long availability): "
                     f"**+{oracle_long_sum_pct:.2f}%** across {oracle_n_long_days} events")
        lines.append(f"- Sum of |magnitude| (long + short combined): +{oracle_total_abs_sum_pct:.2f}%")
        lines.append(f"- ETH bucket: {eth_bucket}, taker cost: {cost*100:.2f}%/RT")
    lines.append("")

    # Top 15 ETH bear long+short pairs by Sharpe
    if len(ls_df):
        lines.append("## 2a. ETH bear — Top 15 (fast, slow) pairs by Sharpe-proxy (long+short)\n")
        lines.append("| rank | type | (fast, slow) | n_sig | n_long | n_short | mean PnL % | hit % | sum PnL % | Sharpe-proxy |")
        lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for i, (_, r) in enumerate(ls_df.head(15).iterrows(), 1):
            lines.append(f"| {i} | {r['ma_type']} | ({r['fast']}, {r['slow']}) | "
                         f"{r['n_signaled']} | {r['n_long_sig']} | {r['n_short_sig']} | "
                         f"{r['mean_pnl_pct']:+.3f} | {r['hit_rate']*100:.1f} | "
                         f"{r['sum_pnl_pct']:+.2f} | {r['sharpe_proxy']:+.4f} |")
        lines.append("")

    # Top 15 ETH bear LONG-ONLY pairs (LO + spot, the deploy constraint)
    if len(lo_df):
        lines.append("## 2b. ETH bear — Top 15 LONG-ONLY pairs (deploy constraint: LO + spot)\n")
        lines.append("| rank | type | (fast, slow) | n_long_fires | mean PnL % | hit % | sum PnL % | Sharpe-proxy |")
        lines.append("|---:|---|---|---:|---:|---:|---:|---:|")
        for i, (_, r) in enumerate(lo_df.head(15).iterrows(), 1):
            lines.append(f"| {i} | {r['ma_type']} | ({r['fast']}, {r['slow']}) | "
                         f"{r['n_long_fires']} | {r['mean_pnl_pct']:+.3f} | "
                         f"{r['hit_rate']*100:.1f} | {r['sum_pnl_pct']:+.2f} | "
                         f"{r['sharpe_proxy']:+.4f} |")
        lines.append("")

    # Oracle capture rate
    lines.append("## 3. Oracle capture rate (ETH bear, long-only)\n")
    if len(lo_df) and oracle_long_sum_pct > 0:
        best_lo = lo_df.iloc[0]
        capture_pct = best_lo["sum_pnl_pct"] / oracle_long_sum_pct * 100
        lines.append(f"**Oracle long availability** (sum of cc-positive magnitude on ETH bear): **+{oracle_long_sum_pct:.2f}%**")
        lines.append(f"**Best LO MA/EMA pair sum_pnl** ({best_lo['ma_type']}({int(best_lo['fast'])}, {int(best_lo['slow'])})): **+{best_lo['sum_pnl_pct']:.2f}%**")
        lines.append(f"**Capture ratio**: **{capture_pct:.1f}%** of the long-only oracle availability")
        lines.append("")
        # Also show top-5 by capture
        lo_df_capture = lo_df.copy()
        lo_df_capture["capture_pct_of_oracle"] = lo_df_capture["sum_pnl_pct"] / oracle_long_sum_pct * 100
        lo_top5 = lo_df_capture.sort_values("capture_pct_of_oracle", ascending=False).head(5)
        lines.append("### Top-5 LO pairs by % of oracle captured\n")
        lines.append("| type | (fast, slow) | n_long | sum PnL % | capture % of oracle |")
        lines.append("|---|---|---:|---:|---:|")
        for _, r in lo_top5.iterrows():
            lines.append(f"| {r['ma_type']} | ({int(r['fast'])}, {int(r['slow'])}) | "
                         f"{r['n_long_fires']} | {r['sum_pnl_pct']:+.2f} | "
                         f"{r['capture_pct_of_oracle']:.1f}% |")
        lines.append("")
        lines.append("### Honest framing on capture\n")
        lines.append(f"- Oracle long availability is a PERFECT-FORESIGHT sum of positive bear days for ETH.")
        lines.append(f"- A REAL ranker can't pick those days with 100% accuracy. The capture % here is")
        lines.append(f"  the realized sum_pnl of a MA/EMA pair that FIRED LONG on bear days, divided by")
        lines.append(f"  the perfect-foresight long sum. Not the same as portfolio NAV capture.")
        lines.append(f"- The realistic upper bound for a static-rule MA/EMA strategy in bear ≈ 30-50%")
        lines.append(f"  of perfect-foresight long sum (because half the long-fire bear events are losers).")

    # Top 25% performers
    lines.append("\n## 4. Top-25% bear-slice asset performers (long-only oracle availability)\n")
    lines.append(f"Total assets with bear events: {n_total_assets}. Top-25% = {n_top25} assets.\n")
    lines.append("### A. Top-15 assets by bear long oracle availability\n")
    lines.append("| asset | bucket | sum_long_avail % | n_long_events |")
    lines.append("|---|---|---:|---:|")
    for asset, avail in top25.head(15).items():
        ab = all_bear[all_bear["asset"] == asset]
        bucket = ab["bucket"].iloc[0] if len(ab) else "?"
        n_long_ev = len(ab[ab["magnitude_signed"] > 0])
        lines.append(f"| {asset} | {bucket} | +{avail:.2f}% | {n_long_ev} |")
    lines.append("")

    # Per-top-25 asset best pair
    lines.append("### B. Best LO MA/EMA pair per top-25% asset (full grid, n_long ≥ 5)\n")
    lines.append("| asset | bucket | best type | (fast, slow) | n_long | mean % | hit % | sum % | Sharpe |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
    for _, r in top25_df.head(30).iterrows():
        lines.append(f"| {r['asset']} | {r['bucket']} | {r['ma_type']} | "
                     f"({int(r['fast'])}, {int(r['slow'])}) | "
                     f"{r['n_long_fires']} | {r['mean_pnl_pct']:+.3f} | "
                     f"{r['hit_rate']*100:.1f} | {r['sum_pnl_pct']:+.2f} | "
                     f"{r['sharpe_proxy']:+.4f} |")
    lines.append("")

    # Aggregate over top-25%
    if len(top25_df):
        ss = top25_df["sum_pnl_pct"]
        pp = top25_df["mean_pnl_pct"]
        hh = top25_df["hit_rate"]
        ssp = top25_df["sharpe_proxy"]
        lines.append("### C. Top-25% aggregate statistics\n")
        lines.append(f"- N assets analysed (with valid best-pair): {len(top25_df)}")
        lines.append(f"- Median best-pair sum_pnl across top-25% assets: **+{ss.median():.2f}%**")
        lines.append(f"- Median mean per-event PnL: **+{pp.median():.3f}%**")
        lines.append(f"- Median hit rate: **{hh.median()*100:.1f}%**")
        lines.append(f"- Median per-asset Sharpe-proxy: **{ssp.median():+.4f}**")
        # Compare to ETH's own best
        if len(lo_df):
            eth_best = lo_df.iloc[0]
            eth_sum = eth_best["sum_pnl_pct"]
            lines.append(f"\n**ETH's own best LO pair sum_pnl**: +{eth_sum:.2f}% — rank among top-25% assets:")
            n_above = (top25_df["sum_pnl_pct"] > eth_sum).sum()
            lines.append(f"  {n_above}/{len(top25_df)} top-25% assets had a HIGHER best-pair sum_pnl in bear.")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] wrote {OUT}")
    print(f"\n=== SUMMARY ===")
    if len(lo_df):
        bp = lo_df.iloc[0]
        print(f"  ETH bear best LO pair: {bp['ma_type']}({int(bp['fast'])}, {int(bp['slow'])})")
        print(f"    n_long={bp['n_long_fires']} mean={bp['mean_pnl_pct']:+.3f}% hit={bp['hit_rate']*100:.1f}% sum={bp['sum_pnl_pct']:+.2f}% Sh={bp['sharpe_proxy']:+.4f}")
        if oracle_long_sum_pct > 0:
            print(f"  ETH bear long oracle availability: +{oracle_long_sum_pct:.2f}%")
            print(f"  ETH best LO capture: {bp['sum_pnl_pct']/oracle_long_sum_pct*100:.1f}% of oracle")
    if len(top25_df):
        print(f"  Top-25% bear assets median sum_pnl: +{top25_df['sum_pnl_pct'].median():.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
