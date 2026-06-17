"""Feature Catalog — metadata reader for config/feature_catalog.yaml.

Companion to feature_registry.py. Distinguished by:
  - feature_registry.yaml  : `sources:` + `chimera_v51:` pipeline-spec schema.
                             Drives the join order. Read by 6 pipeline consumers.
  - feature_catalog.yaml   : `meta:` + `prefix_families:` metadata schema.
                             Per-feature classifications (is_z_scored, preserves_magnitude,
                             is_cross_asset, semantic_class, KS test results, etc.)
                             Read by ML/audit consumers, NOT pipeline consumers.

The split was established 2026-05-20 after commit c59c4e7 accidentally overwrote
the pipeline YAML with catalog data. See src/pipeline/README.md for full provenance.

Public API:
    catalog = FeatureCatalog.load()
    catalog.get_feature("rv_bpv_5m")                 # FeatureMeta dataclass
    catalog.list_by_prefix("xrel_")                  # all xrel_* features
    catalog.find(is_z_scored=True)                   # features that destroy magnitude
    catalog.find(semantic_class="microstructure")
    catalog.summary()                                # counts by prefix / class

Quick CLI:
    python src/pipeline/feature_catalog.py            # print summary
    python src/pipeline/feature_catalog.py --feature norm_funding_momentum
    python src/pipeline/feature_catalog.py --prefix xrel_
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "config" / "feature_catalog.yaml"


@dataclass
class FeatureMeta:
    name: str
    prefix: str
    base_type: str = ""
    semantic_class: str = ""
    description: str = ""
    source_producer: str = ""
    is_z_scored: bool = False
    is_cross_asset: bool = False
    preserves_magnitude: bool = True
    lookahead_safe: bool = True
    expected_range: str = ""
    ks_winner_v_nonmover: float | None = None
    non_null_pct: float | None = None
    notes: str = ""
    added_date: str = ""


@dataclass
class FeatureCatalog:
    version: str = ""
    generated: str = ""
    total_features: int = 0
    features: dict[str, FeatureMeta] = field(default_factory=dict)
    prefix_families: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = DEFAULT_CATALOG_PATH) -> "FeatureCatalog":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"feature catalog not found: {path} "
                f"(expected config/feature_catalog.yaml)")
        with open(path) as f:
            data = yaml.safe_load(f)
        meta = data.get("meta", {}) or {}
        pfx_families = data.get("prefix_families", {}) or {}
        features = {}
        for prefix, family in pfx_families.items():
            family_defaults = {k: v for k, v in family.items() if k != "features"}
            for entry in family.get("features", []) or []:
                fm = FeatureMeta(
                    name=entry["name"],
                    prefix=prefix,
                    base_type=entry.get("base_type", ""),
                    semantic_class=entry.get("semantic_class", family_defaults.get("semantic_class", "")),
                    description=entry.get("description", ""),
                    source_producer=family_defaults.get("source_producer", ""),
                    is_z_scored=bool(family_defaults.get("is_z_scored", False)),
                    is_cross_asset=bool(family_defaults.get("is_cross_asset", False)),
                    preserves_magnitude=bool(family_defaults.get("preserves_magnitude", True)),
                    lookahead_safe=bool(family_defaults.get("lookahead_safe", True)),
                    expected_range=str(entry.get("expected_range", "")),
                    ks_winner_v_nonmover=entry.get("ks_winner_v_nonmover"),
                    non_null_pct=entry.get("non_null_pct"),
                    notes=entry.get("notes", ""),
                    added_date=str(family_defaults.get("added_date", "")),
                )
                features[fm.name] = fm
        return cls(
            version=str(meta.get("version", meta.get("generated", ""))),
            generated=str(meta.get("generated", "")),
            total_features=int(meta.get("total_features", len(features))),
            features=features,
            prefix_families=pfx_families,
        )

    def get_feature(self, name: str) -> FeatureMeta | None:
        return self.features.get(name)

    def list_by_prefix(self, prefix: str) -> list[FeatureMeta]:
        return [f for f in self.features.values() if f.prefix == prefix]

    def find(self, **filters) -> list[FeatureMeta]:
        out = []
        for f in self.features.values():
            if all(getattr(f, k, None) == v for k, v in filters.items()):
                out.append(f)
        return out

    def summary(self) -> dict:
        from collections import Counter
        return {
            "total_features": len(self.features),
            "n_prefixes": len(self.prefix_families),
            "n_z_scored": sum(1 for f in self.features.values() if f.is_z_scored),
            "n_cross_asset": sum(1 for f in self.features.values() if f.is_cross_asset),
            "n_preserves_magnitude": sum(1 for f in self.features.values() if f.preserves_magnitude),
            "by_prefix": dict(Counter(f.prefix for f in self.features.values())),
            "by_semantic_class": dict(Counter(f.semantic_class for f in self.features.values() if f.semantic_class)),
        }


def main():
    ap = argparse.ArgumentParser(description="Feature Catalog inspector")
    ap.add_argument("--feature", help="Print one feature's metadata")
    ap.add_argument("--prefix", help="Print all features under a prefix family")
    ap.add_argument("--z-scored", action="store_true", help="List features that destroy magnitude (norm_*)")
    args = ap.parse_args()

    cat = FeatureCatalog.load()
    if args.feature:
        fm = cat.get_feature(args.feature)
        if fm is None:
            print(f"Feature not found: {args.feature}")
            return 1
        for k, v in fm.__dict__.items():
            print(f"  {k:24s} {v}")
        return 0
    if args.prefix:
        feats = cat.list_by_prefix(args.prefix)
        print(f"Features under prefix '{args.prefix}': {len(feats)}")
        for f in feats:
            print(f"  {f.name:30s}  {f.base_type:14s}  {f.semantic_class:18s}  z={f.is_z_scored} preserves_mag={f.preserves_magnitude}")
        return 0
    if args.z_scored:
        feats = cat.find(is_z_scored=True)
        print(f"Z-scored features (magnitude-destroying): {len(feats)}")
        for f in feats:
            print(f"  {f.name}")
        return 0
    s = cat.summary()
    print("Feature Catalog summary:")
    for k, v in s.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in sorted(v.items(), key=lambda x: -x[1]):
                print(f"    {kk:30s} {vv}")
        else:
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
