"""Gold dataset builder — the model-ready panel per (pair, cadence).

Joins the silver layers into one frame a model can train on directly:
    bars (OHLC + bid/ask + spread + tick_count)
  + per-pair causal features (build_features.py)
  + cross-pair panel broadcast by timestamp (panels.py)
  + universe metadata (dna, pip, price_divisor, tier)
  + TARGETS: forward log returns at multiple horizons (the ONLY non-causal cols,
    explicitly prefixed `fwd_`)
  + split label (train/val/oos/unseen) from the frozen config boundaries, with a
    purge flag on the `purge_bars` rows before each boundary (no normalization leak).

A per-(pair,cadence) lineage manifest (git SHA, source, rows, columns, build time)
is written to data/manifests/.

Output: data/processed/dataset/<cadence>/<pair>_dataset_<YYYYMMDD>.parquet

    python src/pipeline/make_dataset.py --universe u7 --cadences 1h 1d
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import yaml

_PIPELINE_PARENT = str(Path(__file__).resolve().parents[1])
if _PIPELINE_PARENT not in sys.path:
    sys.path.insert(0, _PIPELINE_PARENT)

from pipeline import cli, layout
from pipeline.dispatch import run_per_task
from pipeline.parquet_io import atomic_write_parquet
from pipeline.universe_loader import UniverseLoader

__contract__ = {
    "kind": "pipeline_stage",
    "stage": "make_dataset",
    "inputs": {"upstream": ["bars/<cadence>", "features/<cadence>", "panels/<cadence>"],
               "config_keys": ["splits.*"],
               "args": ["--pairs", "--universe", "--cadences", "--target-horizons"]},
    "outputs": {"files": "data/processed/dataset/<cadence>/<pair>_dataset_<YYYYMMDD>.parquet",
                "manifest": "data/manifests/<PAIR>_dataset_<cadence>.json"},
    "invariants": {"targets_are_fwd_prefixed_only_noncausal": True,
                   "date_based_splits_frozen": True, "purge_gap_applied": True,
                   "atomic_write": True},
    "rationale": "Gold model-ready panel; unseen split reserved for backtest only.",
}

CONFIG_PATH = layout.PROJECT_ROOT / "config" / "data_config.yaml"
TARGET_HORIZONS = (1, 5, 20)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(layout.PROJECT_ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _load_splits() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["splits"]


def _split_expr(splits: dict) -> pl.Expr:
    te = date.fromisoformat(splits["train_end"])
    ve = date.fromisoformat(splits["val_end"])
    oe = date.fromisoformat(splits["oos_end"])
    d = pl.from_epoch(pl.col("timestamp"), time_unit="ms").dt.date()
    return (pl.when(d <= te).then(pl.lit("train"))
            .when(d <= ve).then(pl.lit("val"))
            .when(d <= oe).then(pl.lit("oos"))
            .otherwise(pl.lit("unseen")).alias("split"))


def build_pair(pair: str, cadence: str, splits: dict, git_sha: str, force: bool) -> dict:
    bar_path = layout.bars_latest(pair, cadence)
    if bar_path is None or not bar_path.exists():
        return {"status": "no_bars", "pair": pair, "cadence": cadence}
    bars = pl.read_parquet(bar_path).sort("timestamp")

    feat_path = layout.features_latest(pair, cadence)
    if feat_path and feat_path.exists():
        feat = pl.read_parquet(feat_path)
        if "close" in feat.columns:
            feat = feat.drop("close")
        df = bars.join(feat, on="timestamp", how="left")
    else:
        df = bars

    panel_path = layout.panel_latest("panel", cadence)
    if panel_path and panel_path.exists():
        panel = pl.read_parquet(panel_path)
        df = df.join(panel, on="timestamp", how="left")

    U = UniverseLoader.load()
    df = df.with_columns([
        pl.lit(pair).alias("pair"),
        pl.lit(U.dna_for(pair)).alias("dna"),
        pl.lit(U.pip(pair)).alias("pip"),
        pl.lit(U.price_divisor(pair)).alias("price_divisor"),
    ])

    # carry: join the FRED daily rate panel by date -> rate_base/quote + carry_diff
    # (rates are forward-filled step series; causal).
    rates_files = sorted((layout.PROCESSED / "rates").glob("rates_daily_*.parquet"))
    if rates_files:
        n = layout.normalize_pair(pair)
        rates = pl.read_parquet(rates_files[-1])
        bc, qc = f"rate_{n.base}", f"rate_{n.quote}"
        if bc in rates.columns and qc in rates.columns:
            rs = rates.select(["date", bc, qc]).rename({bc: "rate_base", qc: "rate_quote"})
            df = (df.with_columns(pl.from_epoch("timestamp", "ms").dt.date().alias("_d"))
                  .join(rs, left_on="_d", right_on="date", how="left").drop("_d")
                  .with_columns((pl.col("rate_base") - pl.col("rate_quote")).alias("carry_diff")))

    # targets: forward log returns (future -> non-causal; prefixed fwd_).
    df = df.with_columns([
        (pl.col("close").shift(-h) / pl.col("close")).log().alias(f"fwd_ret_{h}")
        for h in TARGET_HORIZONS
    ])

    # split + purge
    df = df.with_columns(_split_expr(splits))
    purge = int(splits.get("purge_bars", 400))
    next_split = pl.col("split").shift(-purge)
    df = df.with_columns(
        ((pl.col("split") != next_split) & next_split.is_not_null()).alias("is_purge"))

    latest = datetime.fromtimestamp(df["timestamp"].max() / 1000, tz=timezone.utc).date()
    out = layout.dataset_path(pair, cadence, latest_date=latest)
    atomic_write_parquet(df, out, required_cols={"timestamp", "close", "split"})

    # split counts for the summary + manifest
    counts = {r[0]: r[1] for r in df.group_by("split").len().iter_rows()}
    manifest = {
        "pair": pair, "cadence": cadence, "git_sha": git_sha,
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(df), "columns": df.columns,
        "n_features": len([c for c in df.columns
                           if c not in ("timestamp", "pair", "dna", "pip", "price_divisor",
                                        "split", "is_purge")
                           and not c.startswith("fwd_")]),
        "target_horizons": list(TARGET_HORIZONS),
        "split_counts": counts,
        "source_bars": bar_path.name,
    }
    (layout.MANIFESTS).mkdir(parents=True, exist_ok=True)
    (layout.MANIFESTS / f"{layout.normalize_pair(pair).flat}_dataset_{cadence}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return {"status": "ok", "pair": pair, "cadence": cadence, "rows": len(df),
            "cols": len(df.columns), "train": counts.get("train", 0),
            "unseen": counts.get("unseen", 0)}


def _worker(pair, cadence, splits, git_sha, force):
    return build_pair(pair, cadence, splits, git_sha, force)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build gold model-ready FX dataset")
    cli.add_standard_args(ap, default_workers=4, date_window=False)
    ap.add_argument("--cadences", nargs="+", default=["1h", "1d"])
    args = ap.parse_args()
    pairs = cli.resolve_pairs(args, stage_name="make_dataset")
    splits = _load_splits()
    git_sha = _git_sha()
    tasks = [(p, cad, splits, git_sha, args.force) for p in pairs for cad in args.cadences]
    if args.dry_run:
        for t in tasks:
            print(f"  would build dataset {t[0]} {t[1]}")
        return
    run_per_task(tasks, _worker, workers=args.workers,
                 mode="thread" if args.workers > 1 else "serial",
                 stage_name="make_dataset",
                 progress_summary_keys=["cadence", "rows", "cols", "train", "unseen"])


if __name__ == "__main__":
    main()
