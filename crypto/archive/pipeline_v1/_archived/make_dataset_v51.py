"""V51 Dataset Builder — extends V50 chimera with frontier features.

Backward-compatible: V50 chimera files are NOT modified. V51 files are
schema-additive: every V50 column preserved + ~80 frontier features appended.

Pipeline flow:
  Phase 0: Read V50 chimera (already includes 41 features + 10 targets + regime_label).
  Phase 1: For each asset, load silver frontier_features.parquet (built by
           frontier_consolidator.py).
  Phase 2: as-of left-join silver onto V50's dollar-bar timestamp axis (forward-fill
           daily values across ~288 sub-day bars).
  Phase 3: Write V51 chimera + materialize 4h / 1d cadence views.

Run:
  python src/pipeline/make_dataset_v51.py                    # all assets
  python src/pipeline/make_dataset_v51.py --asset BTC        # single asset
  python src/pipeline/make_dataset_v51.py --skip-silver      # assume silver exists
  python src/pipeline/make_dataset_v51.py --no-cadence       # skip 4h/1d materialization

Output:
  data/processed/<sym>_v51_chimera.parquet         # main output (~63+80=143 cols)
  data/processed/<sym>_v51_chimera_1d.parquet      # daily cadence (last bar of day)
  data/processed/<sym>_v51_chimera_4h.parquet      # 4h cadence (last bar of 4h window)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import polars as pl

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from feature_registry import FeatureRegistry  # noqa: E402
from frontier_consolidator import consolidate_one_asset  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def attach_frontier_features(
    chimera_v50: pl.DataFrame,
    frontier_silver: pl.DataFrame,
) -> pl.DataFrame:
    """As-of left-join silver (daily) onto chimera v50 (dollar bars).

    The V50 chimera has a `timestamp` column in ms. Silver has a `date` column.
    Strategy: derive `date` from V50's timestamp, then left-join on date.
    Daily values are forward-filled by definition (silver already filled).
    """
    if "date" not in chimera_v50.columns:
        chimera_v50 = chimera_v50.with_columns([
            pl.from_epoch(pl.col("timestamp"), time_unit="ms").dt.date().alias("date"),
        ])
    silver_no_asset = frontier_silver.drop("asset") if "asset" in frontier_silver.columns else frontier_silver
    out = chimera_v50.join(silver_no_asset, on="date", how="left")
    return out


def materialize_cadence(
    df_v51: pl.DataFrame,
    cadence: str,
) -> pl.DataFrame:
    """Materialize a sub-day cadence view from V51 (which is dollar bars).

    Strategy: for each cadence window, take the LAST bar (close-of-period
    semantics). Sub-day frontier features that were forward-filled keep
    their value; chimera dollar-bar features take the last-bar value.
    """
    if "timestamp" not in df_v51.columns:
        return df_v51
    df = df_v51.with_columns([
        pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("dt"),
    ])
    if cadence == "1d":
        df = df.with_columns(pl.col("dt").dt.date().alias("cadence_key"))
    elif cadence == "4h":
        df = df.with_columns(
            pl.col("dt").dt.truncate("4h").alias("cadence_key")
        )
    elif cadence == "1h":
        df = df.with_columns(pl.col("dt").dt.truncate("1h").alias("cadence_key"))
    elif cadence == "15m":
        df = df.with_columns(pl.col("dt").dt.truncate("15m").alias("cadence_key"))
    else:
        raise ValueError(f"unknown cadence: {cadence}")
    df = df.sort(["cadence_key", "timestamp"]).group_by("cadence_key", maintain_order=True).last()
    df = df.drop(["cadence_key", "dt"])
    return df


def process_asset(
    asset: str,
    registry: FeatureRegistry,
    skip_silver: bool = False,
    do_cadence: bool = True,
) -> dict:
    """Build V51 chimera + cadence views for one asset."""
    asset_l = asset.lower()
    asset_u = asset.upper()
    v50_path = PROCESSED_DIR / f"{asset_l}usdt_v50_chimera.parquet"
    if not v50_path.exists():
        return {"asset": asset_u, "status": "skip_no_v50"}

    silver_path = PROCESSED_DIR / f"{asset_l}usdt_frontier_features.parquet"
    t0 = time.time()

    # Step 1: silver
    if skip_silver and silver_path.exists():
        silver = pl.read_parquet(silver_path)
    else:
        silver = consolidate_one_asset(
            asset_u, registry, out_path=silver_path,
            forward_fill_max_days=registry.chimera.forward_fill_max_days,
        )
        if silver is None or len(silver) == 0:
            return {"asset": asset_u, "status": "no_silver"}

    # Step 2: load v50 + join
    chim_v50 = pl.read_parquet(v50_path)
    n_v50_cols = len(chim_v50.columns)
    chim_v51 = attach_frontier_features(chim_v50, silver)
    n_v51_cols = len(chim_v51.columns)

    # Write
    out_path = PROCESSED_DIR / f"{asset_l}usdt_v51_chimera.parquet"
    chim_v51.write_parquet(out_path, compression="zstd")

    cadence_outputs = {}
    if do_cadence:
        for spec in registry.chimera.cadence_materializations:
            cad_df = materialize_cadence(chim_v51, spec["cadence"])
            cad_path = PROCESSED_DIR / spec["output_pattern"].format(asset_lower=asset_l)
            cad_df.write_parquet(cad_path, compression="zstd")
            cadence_outputs[spec["cadence"]] = {
                "rows": len(cad_df),
                "path": cad_path.name,
            }

    return {
        "asset": asset_u,
        "status": "ok",
        "v50_cols": n_v50_cols,
        "v51_cols": n_v51_cols,
        "added_features": n_v51_cols - n_v50_cols,
        "v51_rows": len(chim_v51),
        "cadence": cadence_outputs,
        "elapsed_s": round(time.time() - t0, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default=None)
    parser.add_argument("--skip-silver", action="store_true")
    parser.add_argument("--no-cadence", action="store_true")
    args = parser.parse_args()

    reg = FeatureRegistry.load()

    if args.asset:
        assets = [args.asset.upper()]
    else:
        v50_files = sorted(PROCESSED_DIR.glob("*_v50_chimera*.parquet"))
        assets = [f.stem.replace("usdt_v50_chimera", "").upper() for f in v50_files]

    print(f"V51 build for {len(assets)} assets, skip_silver={args.skip_silver}, "
          f"cadence={not args.no_cadence}")
    print(f"Expected: 41 base + 80 frontier features = 121 features per asset")
    print()
    results = []
    t_start = time.time()
    for i, asset in enumerate(assets, 1):
        try:
            r = process_asset(asset, reg, skip_silver=args.skip_silver, do_cadence=not args.no_cadence)
        except Exception as e:
            r = {"asset": asset, "status": "error", "err": str(e)}
        results.append(r)
        if r["status"] == "ok":
            cad = ", ".join(f"{c}: {info['rows']}r" for c, info in r["cadence"].items())
            print(f"[{i:>3}/{len(assets)}] {asset:>10} OK   "
                  f"{r['v50_cols']:>3}->{r['v51_cols']:>3} cols  +{r['added_features']:>3} features  "
                  f"{r['v51_rows']:>8} rows  {cad}  ({r['elapsed_s']}s)")
        else:
            print(f"[{i:>3}/{len(assets)}] {asset:>10} {r['status']}  err={r.get('err', r.get('status'))}")
    elapsed = time.time() - t_start
    print()
    print(f"Done. Elapsed {elapsed/60:.1f} min. {sum(1 for r in results if r['status']=='ok')}/{len(assets)} OK.")


if __name__ == "__main__":
    main()
