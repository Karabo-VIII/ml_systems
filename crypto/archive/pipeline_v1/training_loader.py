"""TrainingLoader — canonical training-data API for all model training.

Sits on top of ChimeraLoader. Provides:
  - Train / val / oos / unseen splits via purge_split.py
  - Per-feature normalization stats fit on TRAIN ONLY (no leakage)
  - Cached normalization stats (parquet files) for reuse across model versions
  - Universe filtering (u10 / u50 / u100)
  - Cadence selection (dollar / 1d / 4h / 1h / 15m)
  - Feature subset selection
  - Optional regime-balanced sampling
  - Tensor conversion for torch / numpy

Public API:
    from pipeline.training_loader import TrainingLoader

    tl = TrainingLoader(universe='u10', cadence='dollar',
                        features=['norm_return_1', 'norm_vpin', 'hbr_eta_imbalance'],
                        targets=['target_return_1', 'target_return_4'])
    tl.fit_normalizers()                       # one-time, fit on train only
    train_x, train_y = tl.get_split('train')   # numpy arrays
    val_x, val_y = tl.get_split('val')
    # Or torch:
    loader = tl.torch_dataloader('train', batch_size=32)

Why it matters:
  Every training script needs train/val/oos/unseen + normalized inputs + matched
  targets. Without a canonical API, every script reimplements this ~150 LOC of
  glue. With drift between scripts, models train on subtly different data
  distributions. TrainingLoader is the single source of truth.
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from chimera_loader import ChimeraLoader  # noqa: E402
from purge_split import split_chimera, get_split_dates  # noqa: E402
from universe_loader import UniverseLoader  # noqa: E402
from parquet_io import atomic_write_parquet  # noqa: E402


def _stable_feat_hash(features) -> str:
    """Deterministic short hash of the feature set (stable across processes).

    Python's builtin hash() is salted per-process (PYTHONHASHSEED), so it would
    produce a different normalization-cache filename on every run, defeating the
    cache. Use a content hash instead.
    """
    if not features:
        return "0"
    joined = ",".join(sorted(features)).encode("utf-8")
    return hashlib.sha1(joined).hexdigest()[:12]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
NORM_CACHE_DIR = PROJECT_ROOT / "data" / "_normalizers"


@dataclass
class NormStats:
    feature: str
    mean: float
    std: float
    n: int


@dataclass
class TrainingLoader:
    universe: str = "u10"
    cadence: str = "dollar"
    features: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=lambda: ["target_return_1"])
    asset_subset: list[str] | None = None  # override universe; specific symbols
    normalize: bool = True
    drop_nan_targets: bool = True
    cache_key: str | None = None  # name for normalization cache file

    _chimera: ChimeraLoader = field(default=None, init=False, repr=False)
    _universes: UniverseLoader = field(default=None, init=False, repr=False)
    _norm_stats: dict[str, NormStats] = field(default_factory=dict, init=False, repr=False)
    _cached_panel: pl.DataFrame | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._chimera = ChimeraLoader()
        self._universes = UniverseLoader.load()
        if not self.cache_key:
            feats_hash = _stable_feat_hash(self.features)
            self.cache_key = f"{self.universe}_{self.cadence}_h{feats_hash}"

    def list_assets(self) -> list[str]:
        if self.asset_subset:
            return [s.upper() if s.upper().endswith("USDT") else s.upper() + "USDT"
                    for s in self.asset_subset]
        return self._universes.list(self.universe)

    def _load_full_panel(self) -> pl.DataFrame:
        """Load and cache the full universe panel (all assets, all dates, requested features+targets)."""
        if self._cached_panel is not None:
            return self._cached_panel
        # Ordered dedupe (preserve requested feature/target order; set() scrambles).
        _seen: set = set()
        cols = [c for c in (self.features + self.targets + ["timestamp"])
                if not (c in _seen or _seen.add(c))]
        # If the symbol-specific load fails, surface; we want loud errors at training time.
        all_cols = cols
        if self.asset_subset:
            frames = []
            for sym in self.list_assets():
                try:
                    df = self._chimera.load(sym, cadence=self.cadence, features=all_cols)
                    df = df.with_columns(pl.lit(sym).alias("asset"))
                    frames.append(df)
                except FileNotFoundError:
                    continue
            if not frames:
                raise FileNotFoundError(f"No assets in {self.list_assets()} have v51 chimera")
            self._cached_panel = pl.concat(frames, how="vertical_relaxed")
        else:
            self._cached_panel = self._chimera.load_universe(
                self.universe, cadence=self.cadence,
                features=all_cols, add_asset_col=True, skip_missing=True,
            )
        return self._cached_panel

    def fit_normalizers(self, force: bool = False) -> dict[str, NormStats]:
        """Fit per-feature mean/std on TRAIN segment only. Cache to disk.

        For features with no train coverage (e.g., ETF flows pre-2024), uses
        ALL available data (train+val+oos) to fit stats. Surfaces a warning
        per such feature so caller knows the train-only-no-leakage rule was
        relaxed for that column.
        """
        if not self.normalize:
            return {}
        cache_path = NORM_CACHE_DIR / f"{self.cache_key}.parquet"
        if cache_path.exists() and not force:
            df = pl.read_parquet(cache_path)
            self._norm_stats = {
                row["feature"]: NormStats(row["feature"], row["mean"], row["std"], row["n"])
                for row in df.to_dicts()
            }
            return self._norm_stats

        panel = self._load_full_panel()
        train, val, oos, unseen = split_chimera(panel)
        rows = []
        for f in self.features:
            if f not in panel.columns:
                print(f"[tl] WARN: feature '{f}' not in chimera; skipped")
                continue
            s_train = train[f].drop_nulls() if f in train.columns else pl.Series([])
            if len(s_train) >= 100:
                source = "train"
                s = s_train
            else:
                # Fall back to all-data normalization with warning
                s = panel[f].drop_nulls()
                if len(s) < 100:
                    print(f"[tl] WARN: feature '{f}' has <100 non-null obs total; skipped")
                    continue
                source = "all_data_fallback"
                print(f"[tl] WARN: feature '{f}' has {len(s_train)} train obs; "
                      f"falling back to all-data normalization (n={len(s)})")
            mu = float(s.mean() or 0.0)
            sigma = float(s.std() or 1.0)
            if sigma < 1e-9:
                sigma = 1.0
            rows.append({"feature": f, "mean": mu, "std": sigma, "n": len(s), "source": source})
            self._norm_stats[f] = NormStats(f, mu, sigma, len(s))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if rows:
            # Atomic write so a crash mid-write can't leave a corrupt cache that
            # the next run reads as valid stats.
            atomic_write_parquet(pl.DataFrame(rows), cache_path)
        return self._norm_stats

    def _apply_normalization(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.normalize or not self._norm_stats:
            return df
        exprs = []
        for f, ns in self._norm_stats.items():
            if f in df.columns:
                exprs.append(((pl.col(f) - ns.mean) / ns.std).alias(f))
        if exprs:
            df = df.with_columns(exprs)
        return df

    def get_split(self, segment: str = "train") -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
        """Return (X, y, meta_df). X is normalized features, y is targets, meta is the polars df."""
        # Guard: if normalization is on but stats were never fit, fit now rather
        # than silently returning un-normalized features.
        if self.normalize and not self._norm_stats:
            self.fit_normalizers()
        panel = self._load_full_panel()
        train, val, oos, unseen = split_chimera(panel)
        seg = {"train": train, "val": val, "oos": oos, "unseen": unseen}.get(segment)
        if seg is None:
            raise ValueError(f"unknown segment: {segment}; use train|val|oos|unseen")
        if self.drop_nan_targets:
            for tg in self.targets:
                if tg in seg.columns:
                    seg = seg.drop_nulls(tg)
        seg_normed = self._apply_normalization(seg)
        feats_present = [f for f in self.features if f in seg_normed.columns]
        targets_present = [t for t in self.targets if t in seg_normed.columns]
        # Fill NaN in X with 0 (neutral / missing-observation assumption).
        # Models that need explicit missingness can mask zeros.
        X = seg_normed.select([
            pl.col(f).fill_null(0.0).fill_nan(0.0).alias(f) for f in feats_present
        ]).to_numpy()
        y = seg_normed.select(targets_present).to_numpy()
        return X, y, seg

    def summary(self) -> dict:
        """Return shape + segment sizes (cheap pre-train sanity)."""
        panel = self._load_full_panel()
        train, val, oos, unseen = split_chimera(panel)
        return {
            "universe": self.universe,
            "cadence": self.cadence,
            "n_features": len(self.features),
            "n_targets": len(self.targets),
            "panel_rows": len(panel),
            "panel_assets": panel["asset"].n_unique() if "asset" in panel.columns else None,
            "train_rows": len(train),
            "val_rows": len(val),
            "oos_rows": len(oos),
            "unseen_rows": len(unseen),
            "norm_cache_key": self.cache_key,
        }


def main():
    """CLI smoke test."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="u10")
    ap.add_argument("--cadence", default="1d")
    ap.add_argument("--features", default="norm_return_1,norm_vpin,etf_btc_etf_total_z30")
    ap.add_argument("--targets", default="target_return_1")
    args = ap.parse_args()

    tl = TrainingLoader(
        universe=args.universe,
        cadence=args.cadence,
        features=[f.strip() for f in args.features.split(",")],
        targets=[t.strip() for t in args.targets.split(",")],
    )
    print("[tl] config:", tl.universe, tl.cadence)
    print("[tl] summary:", tl.summary())
    print("[tl] fitting normalizers on train...")
    stats = tl.fit_normalizers()
    for f, ns in list(stats.items())[:5]:
        print(f"  {f}: mu={ns.mean:.4f} sigma={ns.std:.4f} n={ns.n}")
    Xt, yt, _ = tl.get_split("train")
    Xv, yv, _ = tl.get_split("val")
    print(f"[tl] train X={Xt.shape} y={yt.shape}; val X={Xv.shape} y={yv.shape}")
    print(f"[tl] X mean={Xt.mean():.4f}, std={Xt.std():.4f} (post-normalization)")


if __name__ == "__main__":
    main()
