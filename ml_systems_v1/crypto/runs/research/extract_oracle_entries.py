"""extract_oracle_entries.py -- per-(asset,cadence) ORACLE-ENTRIES table from the capturable catalogs.

WHAT
----
Emits, for every asset present in the capturable catalogs, the perfect-foresight LONG-ONLY
NON-OVERLAPPING set of oracle ENTRY timestamps (the "setup across a multi-candle move" the
oracle exploits). This is Step-1 ("CONSTRUCT THE ORACLE") of docs/ORACLE_DECOMPOSITION_2026_06_06.md,
producing the entry table that Step-2 (DNA decomposition) consumes as its bar labels.

WHY NOT JUST THE CATALOG WIN-FLAGS
----------------------------------
The catalog win-flags (win_5pct_18bar etc.) are DENSE candidate labels: ~33-41% of bars carry a
flag because forward windows overlap. Empirically (verified in this task) the win-run START bar is
NOT the max-capture entry -- later bars in a run have higher forward gain because each bar's window
slides forward to a later/bigger peak. So the flags are INSUFFICIENT to recover clean, non-
overlapping, max-capture entries. We therefore build the DP oracle (the catalog-insufficient branch
the task allows), reusing the exact self-tested DP in oracle_ceiling_builder.py.

ORACLE CONTRACT (matches oracle_ceiling_builder.py, run CLOSE-to-CLOSE on the catalog price path)
  * entry = close[i], exit = close[j], j > i        (close-to-close; conservative -- no intrabar-high optimism)
  * hold-time band: >= 1h (>=1 bar) and < 7 days     (per real ts; doc spec)
  * NON-OVERLAPPING single position (next entry >= exit + 1)
  * net round-trip TAKER cost = 0.0024 subtracted per move
  * objective = maximise COMPOUND wealth (product of per-move multipliers); skip dominates losers
  * the DP only ever selects net-positive up-legs -> every entry is a captured move.

NOTE the canonical oracle_ceiling_map.json uses entry=open / exit=HIGH (a looser, optimistic ceiling).
This table is the CLOSE-to-close (honest/lower) sibling, grounded directly in the named catalogs which
carry close+ts for all 87 assets. The oracle is clairvoyant by construction -- used ONLY to DEFINE the
target, never as a feature (per the doc).

OUTPUT (runs/research/)
  oracle_entries_4h.parquet      long: one row per oracle move, 4h cadence, all assets
  oracle_entries_daily.parquet   long: one row per oracle move, daily cadence, all assets
  oracle_entries_summary.parquet per-(asset,cadence) rollup
  oracle_entries_report.json     headline numbers + provenance

RWYB:  .venv/Scripts/python.exe runs/research/extract_oracle_entries.py --selftest
       .venv/Scripts/python.exe runs/research/extract_oracle_entries.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runs" / "research"))
sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Reuse the EXACT self-tested DP + constants (do not reinvent).
from oracle_ceiling_builder import (  # noqa: E402
    COST_RT,
    MIN_HOLD_HOURS,
    MS_PER_HOUR,
    oracle_high_capture,
    _selftest as _dp_selftest,
)

CADENCES = {
    "4h": {
        "path": ROOT / "data" / "processed" / "capturable_4h_catalog.parquet",
        "tcol": "ts",
    },
    "daily": {
        "path": ROOT / "data" / "processed" / "capturable_win_catalog.parquet",
        "tcol": "date",
    },
}


def _ts_ms_from_datetime(series: pl.Series) -> np.ndarray:
    """polars Datetime(ns) -> int64 epoch-ms."""
    # cast to int gives ns since epoch; //1e6 -> ms
    ns = series.cast(pl.Int64).to_numpy()
    return (ns // 1_000_000).astype(np.int64)


def extract_for_cadence(cad: str, cfg: dict) -> tuple[pl.DataFrame, list[dict]]:
    """Run the close-to-close oracle DP per asset; return (entries_df, per_asset_summaries)."""
    tcol = cfg["tcol"]
    df = pl.read_parquet(cfg["path"]).sort(["asset", tcol])
    assets = df["asset"].unique().to_list()
    assets.sort()
    rows = []
    summaries = []
    for a in assets:
        sub = df.filter(pl.col("asset") == a).sort(tcol)
        ts_ms = _ts_ms_from_datetime(sub[tcol])
        close = sub["close"].to_numpy().astype(np.float64)
        split = sub["split"].to_numpy()
        n = len(close)
        # guard: strictly increasing ts (searchsorted contract). dedupe by ts (keep first).
        if n == 0:
            continue
        if not np.all(np.diff(ts_ms) > 0):
            keep = np.ones(n, dtype=bool)
            keep[1:] = np.diff(ts_ms) > 0
            ts_ms, close, split = ts_ms[keep], close[keep], split[keep]
            n = len(close)
        if n < 3 or not np.all(np.isfinite(close)) or np.any(close <= 0):
            summaries.append({"asset": a, "cadence": cad, "n_bars": n, "n_moves": 0,
                              "skipped": "too_short_or_bad_close"})
            continue
        # close-to-close oracle = high-capture DP with open_=high=close.
        f, trades = oracle_high_capture(ts_ms, close, close, cost=COST_RT,
                                        min_hold_hours=MIN_HOLD_HOURS)
        nets = []
        for (i, j) in trades:
            gross = close[j] / close[i] - 1.0
            net = gross - COST_RT
            hold_h = (ts_ms[j] - ts_ms[i]) / MS_PER_HOUR
            rows.append({
                "asset": a,
                "cadence": cad,
                "entry_ts_ms": int(ts_ms[i]),
                "exit_ts_ms": int(ts_ms[j]),
                "entry_close": float(close[i]),
                "exit_close": float(close[j]),
                "hold_hours": float(hold_h),
                "hold_bars": int(j - i),
                "gross_ret": float(gross),
                "net_ret": float(net),
                "split": str(split[i]),
            })
            nets.append(net)
        nets = np.array(nets) if nets else np.array([])
        comp = float(np.prod(1.0 + nets) - 1.0) if len(nets) else 0.0
        span_days = float((ts_ms[-1] - ts_ms[0]) / (24 * 3600 * 1000))
        summaries.append({
            "asset": a, "cadence": cad, "n_bars": int(n), "n_moves": int(len(trades)),
            "span_days": round(span_days, 1),
            "moves_per_year": round(len(trades) / (span_days / 365.25), 2) if span_days > 0 else 0.0,
            "mean_net_per_move_pct": round(float(nets.mean() * 100), 4) if len(nets) else 0.0,
            "median_net_per_move_pct": round(float(np.median(nets) * 100), 4) if len(nets) else 0.0,
            "oracle_compound_pct": round(comp * 100, 2),
            "dp_wealth_mult": round(float(f[0]), 6),
        })
    entries_df = pl.DataFrame(rows) if rows else pl.DataFrame()
    # attach human-readable entry timestamp
    if entries_df.height:
        entries_df = entries_df.with_columns([
            pl.from_epoch(pl.col("entry_ts_ms"), time_unit="ms").alias("entry_ts"),
            pl.from_epoch(pl.col("exit_ts_ms"), time_unit="ms").alias("exit_ts"),
        ])
    return entries_df, summaries


def main():
    print("=" * 78)
    print("ORACLE-ENTRIES EXTRACTION  (close-to-close DP on capturable catalogs)")
    print(f"cost_rt={COST_RT}  min_hold_h={MIN_HOLD_HOURS}  max_hold<7d  objective=max-compound")
    print("=" * 78)
    report = {"cost_rt": COST_RT, "min_hold_hours": MIN_HOLD_HOURS, "max_hold_days": 7,
              "method": "close-to-close perfect-foresight non-overlap DP (reuses oracle_ceiling_builder DP)",
              "note": "CLOSE-based (honest/lower) sibling of high-based oracle_ceiling_map.json",
              "cadences": {}}
    all_summ = []
    for cad, cfg in CADENCES.items():
        if not cfg["path"].exists():
            print(f"[SKIP] {cad}: missing {cfg['path']}")
            continue
        edf, summ = extract_for_cadence(cad, cfg)
        all_summ.extend(summ)
        outp = ROOT / "runs" / "research" / f"oracle_entries_{cad}.parquet"
        if edf.height:
            edf.write_parquet(outp)
        n_assets = len({s["asset"] for s in summ})
        n_moves = int(edf.height)
        comps = [s["oracle_compound_pct"] for s in summ if s.get("n_moves", 0) > 0]
        mpy = [s["moves_per_year"] for s in summ if s.get("n_moves", 0) > 0]
        mnet = [s["mean_net_per_move_pct"] for s in summ if s.get("n_moves", 0) > 0]
        report["cadences"][cad] = {
            "n_assets": n_assets,
            "total_oracle_moves": n_moves,
            "median_moves_per_year": round(float(np.median(mpy)), 2) if mpy else 0.0,
            "median_mean_net_per_move_pct": round(float(np.median(mnet)), 3) if mnet else 0.0,
            "median_oracle_compound_pct": round(float(np.median(comps)), 1) if comps else 0.0,
            "output": str(outp.relative_to(ROOT)) if edf.height else None,
        }
        print(f"\n[{cad}] assets={n_assets}  oracle_moves={n_moves}  "
              f"median moves/yr={report['cadences'][cad]['median_moves_per_year']}  "
              f"median net/move={report['cadences'][cad]['median_mean_net_per_move_pct']}%  "
              f"median oracle compound={report['cadences'][cad]['median_oracle_compound_pct']}%")
        # show a few assets
        shown = [s for s in summ if s.get("n_moves", 0) > 0][:6]
        for s in shown:
            print(f"    {s['asset']:8} n_bars={s['n_bars']:>6} moves={s['n_moves']:>5} "
                  f"mv/yr={s['moves_per_year']:>6} net/move={s['mean_net_per_move_pct']:>6.2f}% "
                  f"compound={s['oracle_compound_pct']:>12.1f}%")

    summ_df = pl.DataFrame(all_summ)
    if summ_df.height:
        summ_df.write_parquet(ROOT / "runs" / "research" / "oracle_entries_summary.parquet")
    (ROOT / "runs" / "research" / "oracle_entries_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print("\n[OK] wrote oracle_entries_{4h,daily}.parquet + _summary.parquet + _report.json")
    return report


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        ok = _dp_selftest()
        sys.exit(0 if ok else 1)
    print("[pre-flight] running imported DP self-test...")
    if not _dp_selftest():
        print("DP SELF-TEST FAILED -- aborting")
        sys.exit(1)
    print()
    main()
