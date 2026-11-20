"""materialize_oracle_entries_u100.py -- per-asset ORACLE-ENTRY tables from the
capturable catalogs, restricted to u100 (2026-06-06, RWYB).

TASK
----
"Load capturable_4h_catalog.parquet + capturable_win_catalog.parquet to materialize
oracle entries (capturable multi-candle moves) per asset across u100; only build the
DP oracle if a needed cadence is missing from the catalogs. Emit per-asset
oracle-entry tables with timestamps + realized forward move."

WHAT AN "ORACLE ENTRY" IS HERE
------------------------------
The catalogs are clean-win labeled per the builders
(archive/scripts/oracle/build_4h_capturable_catalog.py,
 archive/scripts/oracle/build_capturable_win_catalog.py):

  4h   : win_3pct_12bar = 1  iff  max_gain_12bar >= 0.03 AND max_loss_12bar > -0.05
         win_5pct_18bar = 1  iff  max_gain_18bar >= 0.05 AND max_loss_18bar > -0.05
  daily: win_Xpct       = 1  iff  max_gain_3d    >= X    AND max_loss_3d    > -0.05

i.e. a clean-win bar = a bar from which a LONG entry at close[t] reaches a +X%
capturable move within the forward window WITHOUT the intrabar path first breaching
-5% (conservative path-capturability under OHLC granularity). That clean-win bar IS
the oracle entry; the realized forward move = max_gain (max favorable excursion = the
capturable ceiling, exit-at-peak). max_loss is the worst drawdown along the path.

CADENCE COVERAGE -> NO DP BUILD
-------------------------------
The two needed cadences are {4h (capturable_4h_catalog), daily (capturable_win_catalog)}.
BOTH are present, so per the task ("only build the DP oracle if a needed cadence is
missing") the DP oracle is NOT built here. (A stricter perfect-foresight non-overlap
DP variant already exists at runs/research/extract_oracle_entries.py and
runs/research/oracle_dp_uncovered_cadences.py for the uncovered {1h,30m,15m,dollar}.)

OUTPUT (runs/research/)
  oracle_entries_catalog_4h_u100.parquet      long: 1 row per (asset, entry bar, move_def), 4h
  oracle_entries_catalog_daily_u100.parquet   long: 1 row per (asset, entry bar, move_def), daily
  oracle_entries_catalog_summary_u100.parquet per-(asset,cadence,move_def) rollup
  oracle_entries_catalog_report_u100.json     headline numbers + provenance + u100 audit

Per-asset extraction = df.filter(pl.col('asset')==SYM); the 'asset' column is the partition.

RWYB:  python runs/research/materialize_oracle_entries_u100.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from pipeline.universe_loader import UniverseLoader  # noqa: E402

COST_RT = 0.0024            # net round-trip taker cost (project canonical)
SL = -0.05                  # the -5% path-capturability stop baked into the catalogs

CAT_4H = ROOT / "data" / "processed" / "capturable_4h_catalog.parquet"
CAT_DAILY = ROOT / "data" / "processed" / "capturable_win_catalog.parquet"

# move_def -> (win_flag_col, mfe_col, mae_col, horizon_bars, horizon_hours)
DEFS_4H = {
    "3pct_12bar": ("win_3pct_12bar", "max_gain_12bar", "max_loss_12bar", 12, 48),
    "5pct_18bar": ("win_5pct_18bar", "max_gain_18bar", "max_loss_18bar", 18, 72),
}
DEFS_DAILY = {
    "3pct_3d":  ("win_3pct",  "max_gain_3d", "max_loss_3d", 3, 72),
    "5pct_3d":  ("win_5pct",  "max_gain_3d", "max_loss_3d", 3, 72),
    "7pct_3d":  ("win_7pct",  "max_gain_3d", "max_loss_3d", 3, 72),
    "10pct_3d": ("win_10pct", "max_gain_3d", "max_loss_3d", 3, 72),
}


def u100_base_symbols() -> tuple[set[str], list[str]]:
    U = UniverseLoader.load()
    full = U.list("u100")
    base = {s[:-4] if s.endswith("USDT") else s for s in full}
    return base, sorted(full)


def materialize(cat_path: Path, tcol: str, defs: dict, cadence: str,
                u100_base: set[str]) -> tuple[pl.DataFrame, list[dict], dict]:
    df = pl.read_parquet(cat_path)
    cat_assets = set(df["asset"].unique().to_list())
    in_u100 = sorted(cat_assets & u100_base)
    dropped_non_u100 = sorted(cat_assets - u100_base)
    df = df.filter(pl.col("asset").is_in(in_u100)).sort(["asset", tcol])

    # per-asset bar counts (denominator for base rates)
    bars_per_asset = {r["asset"]: r["len"]
                      for r in df.group_by("asset").len().to_dicts()}

    entry_frames = []
    for move_def, (wcol, mfe, mae, hb, hh) in defs.items():
        sub = (df.filter(pl.col(wcol) == 1)
                 .select([
                     pl.col("asset"),
                     pl.lit(cadence).alias("cadence"),
                     pl.lit(move_def).alias("move_def"),
                     pl.col(tcol).alias("entry_ts"),
                     pl.col("close").alias("entry_close"),
                     pl.col(mfe).alias("realized_fwd_mfe"),
                     (pl.col(mfe) - COST_RT).alias("net_fwd_move"),
                     pl.col(mae).alias("realized_fwd_mae"),
                     pl.lit(hb).alias("horizon_bars"),
                     pl.lit(hh).alias("horizon_hours"),
                     pl.col("split"),
                 ]))
        entry_frames.append(sub)
    entries = pl.concat(entry_frames, how="vertical").sort(["asset", "move_def", "entry_ts"])

    # per-(asset, move_def) summary
    summaries = []
    for move_def, (wcol, mfe, mae, hb, hh) in defs.items():
        e = entries.filter(pl.col("move_def") == move_def)
        for a in in_u100:
            ea = e.filter(pl.col("asset") == a)
            nbars = bars_per_asset.get(a, 0)
            ne = ea.height
            if ne == 0:
                summaries.append({"asset": a, "cadence": cadence, "move_def": move_def,
                                  "n_bars": nbars, "n_entries": 0, "base_rate": 0.0})
                continue
            mfe_arr = ea["realized_fwd_mfe"].to_numpy()
            net_arr = ea["net_fwd_move"].to_numpy()
            ts = ea["entry_ts"]
            span_days = (ts.max() - ts.min()).total_seconds() / 86400.0
            summaries.append({
                "asset": a, "cadence": cadence, "move_def": move_def,
                "n_bars": int(nbars), "n_entries": int(ne),
                "base_rate": round(ne / nbars, 4) if nbars else 0.0,
                "span_days": round(span_days, 1),
                "entries_per_year": round(ne / (span_days / 365.25), 1) if span_days > 0 else 0.0,
                "mean_mfe_pct": round(float(mfe_arr.mean()) * 100, 3),
                "median_mfe_pct": round(float(np.median(mfe_arr)) * 100, 3),
                "mean_net_pct": round(float(net_arr.mean()) * 100, 3),
                "max_mfe_pct": round(float(mfe_arr.max()) * 100, 2),
            })

    audit = {
        "cadence": cadence, "catalog": str(cat_path.relative_to(ROOT)),
        "n_catalog_assets": len(cat_assets),
        "n_u100_assets_materialized": len(in_u100),
        "n_dropped_non_u100": len(dropped_non_u100),
        "dropped_non_u100": dropped_non_u100,
        "total_oracle_entries": int(entries.height),
        "move_defs": {k: {"win_flag": v[0], "mfe_col": v[1], "mae_col": v[2],
                          "horizon_bars": v[3], "horizon_hours": v[4]}
                      for k, v in defs.items()},
    }
    return entries, summaries, audit


def main():
    print("=" * 80)
    print("MATERIALIZE ORACLE ENTRIES (capturable multi-candle moves) FROM CATALOGS -> u100")
    print(f"cost_rt={COST_RT}  path-stop baked into clean-win flags = {SL}")
    print("=" * 80)

    for p in (CAT_4H, CAT_DAILY):
        if not p.exists():
            print(f"[FATAL] missing catalog: {p}")
            return 2

    u100_base, u100_full = u100_base_symbols()
    print(f"u100: {len(u100_full)} symbols ({len(u100_base)} base)")

    report = {
        "task": "materialize per-asset oracle entries (capturable multi-candle moves) across u100",
        "cost_rt": COST_RT, "path_stop_in_clean_win": SL,
        "oracle_entry_definition": (
            "clean-win bar = LONG entry at close[t] reaches +X% capturable move within the "
            "forward window WITHOUT the path first breaching -5% (path-capturability under "
            "OHLC granularity). realized_fwd_mfe = max favorable excursion = capturable ceiling "
            "(exit-at-peak); net_fwd_move = mfe - cost_rt; realized_fwd_mae = worst drawdown."),
        "cadence_coverage": {
            "needed": ["4h", "daily"],
            "covered_by_catalogs": ["4h", "daily"],
            "missing": [],
            "dp_oracle_built": False,
            "dp_note": ("both needed cadences present -> DP oracle NOT built (task conditional "
                        "not triggered). Stricter non-overlap DP variant exists at "
                        "runs/research/extract_oracle_entries.py for uncovered cadences."),
        },
        "u100_n_symbols": len(u100_full),
        "cadences": {},
    }

    all_summ = []
    for cadence, cat_path, tcol, defs in [
        ("4h", CAT_4H, "ts", DEFS_4H),
        ("daily", CAT_DAILY, "date", DEFS_DAILY),
    ]:
        entries, summ, audit = materialize(cat_path, tcol, defs, cadence, u100_base)
        all_summ.extend(summ)
        outp = ROOT / "runs" / "research" / f"oracle_entries_catalog_{cadence}_u100.parquet"
        entries.write_parquet(outp)
        audit["output"] = str(outp.relative_to(ROOT))
        report["cadences"][cadence] = audit

        print(f"\n[{cadence}] catalog_assets={audit['n_catalog_assets']} "
              f"u100_materialized={audit['n_u100_assets_materialized']} "
              f"dropped_non_u100={audit['n_dropped_non_u100']}")
        print(f"  total oracle entries (all move_defs): {audit['total_oracle_entries']:,}")
        # per move_def headline
        for move_def in defs:
            ms = [s for s in summ if s["move_def"] == move_def and s["n_entries"] > 0]
            if not ms:
                continue
            tot = sum(s["n_entries"] for s in ms)
            med_base = float(np.median([s["base_rate"] for s in ms]))
            med_mfe = float(np.median([s["mean_mfe_pct"] for s in ms]))
            med_epy = float(np.median([s["entries_per_year"] for s in ms]))
            print(f"    {move_def:11} entries={tot:>7,}  assets={len(ms):>3}  "
                  f"median base_rate={med_base:.3f}  median mean_MFE={med_mfe:.2f}%  "
                  f"median entries/yr={med_epy:.0f}")
        print(f"  -> {outp.relative_to(ROOT)}")

    summ_df = pl.DataFrame(all_summ)
    summ_df.write_parquet(ROOT / "runs" / "research" / "oracle_entries_catalog_summary_u100.parquet")
    (ROOT / "runs" / "research" / "oracle_entries_catalog_report_u100.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\n[OK] wrote oracle_entries_catalog_{4h,daily}_u100.parquet + _summary_u100.parquet + _report_u100.json")

    # show a few example per-asset rows (proof of timestamps + realized move)
    ex = pl.read_parquet(ROOT / "runs" / "research" / "oracle_entries_catalog_4h_u100.parquet")
    print("\n[example] first 4 BTC 4h oracle entries (move_def=5pct_18bar):")
    btc = ex.filter((pl.col("asset") == "BTC") & (pl.col("move_def") == "5pct_18bar")).head(4)
    for r in btc.to_dicts():
        print(f"    {r['entry_ts']}  close={r['entry_close']:.1f}  "
              f"MFE={r['realized_fwd_mfe']*100:.2f}%  net={r['net_fwd_move']*100:.2f}%  "
              f"MAE={r['realized_fwd_mae']*100:.2f}%  split={r['split']}")
    return report


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
