"""Feature Registry — declarative spec loader for chimera v51.

Single source of truth for "what features exist, where they come from,
what cadence." Backed by config/feature_registry.yaml.

Public API:
    registry = FeatureRegistry.load()
    registry.list_sources()                 # ['hawkes_branching', ...]
    registry.list_features()                # all feature names with prefixes
    registry.get_source(name)               # SourceSpec dataclass
    registry.get_chimera_join_order()       # list of source names in join order
    registry.get_expected_v51_features()    # int -- expected total feature count

Adding a new feature: edit config/feature_registry.yaml, no code change needed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "config" / "feature_registry.yaml"


@dataclass
class SourceSpec:
    name: str
    path: str
    layout: Literal["per_asset", "global", "wide_per_asset"]
    date_col: str
    date_unit: Literal["date", "datetime"]
    features: list[str] = field(default_factory=list)
    asset_col: str | None = None
    wide_pattern: str | None = None
    feature_alias: str | None = None
    prefix: str = ""
    min_date: str | None = None
    notes: str = ""
    expected_coverage: dict | None = None   # {'universe': 'u10' | 'u50'} or {'assets': [...]}

    def absolute_path(self) -> Path:
        """Resolve to a file on disk.

        Path in registry is relative to data/. Post-2026-04-26: panel files in
        processed/{hawkes,panels}/ may carry a `_<YYYYMMDD>` suffix; glob for
        latest if the exact path doesn't exist.
        """
        p = PROJECT_ROOT / "data" / self.path
        if p.exists():
            return p
        # Try latest-dated fallback in same dir with same basename stem
        directory = p.parent
        if directory.exists():
            stem = p.stem
            candidates = sorted(directory.glob(f"{stem}_*.parquet"))
            dated = []
            for c in candidates:
                tail = c.stem.rsplit("_", 1)[-1]
                if tail.isdigit() and len(tail) == 8:
                    dated.append(c)
            if dated:
                return dated[-1]
        return p  # caller may .exists() check

    def output_feature_names(self) -> list[str]:
        """Feature names as they appear in chimera v51 (with prefix)."""
        if self.layout == "wide_per_asset":
            base = self.feature_alias or "value"
            return [f"{self.prefix}{base}"]
        return [f"{self.prefix}{f}" for f in self.features]


@dataclass
class ChimeraSpec:
    base_source: str
    base_pattern: str
    output_pattern: str
    sources_to_join: list[str]
    missing_value_policy: Literal["forward_fill", "leave_nan"]
    forward_fill_max_days: int
    expected_new_features_per_source: dict[str, int]
    expected_total_new_features: int
    expected_chimera_v51_total_features: int
    cadence_materializations: list[dict] = field(default_factory=list)
    # 2026-05-24: T2-A surgical Phase 2 added bar-grain features that are
    # NOT in sources_to_join (which is daily-silver only). Tracked separately
    # so cardinality checks can validate the total = frontier_sum + bargrain.
    expected_bargrain_features: int = 0


@dataclass
class FeatureRegistry:
    version: str
    sources: dict[str, SourceSpec]
    chimera: ChimeraSpec

    @classmethod
    def load(cls, path: Path = DEFAULT_REGISTRY_PATH) -> "FeatureRegistry":
        import dataclasses as _dc
        with open(path) as f:
            data = yaml.safe_load(f)

        def _filter_known(cls_, spec_dict, ctx):
            # Drop YAML keys not in the dataclass (e.g. doc-only annotations) so a
            # benign extra key doesn't raise TypeError and break the whole loader.
            known = {fld.name for fld in _dc.fields(cls_)}
            extra = set(spec_dict) - known
            if extra:
                print(f"[feature_registry] WARN {ctx}: ignoring unknown registry "
                      f"key(s) {sorted(extra)}", flush=True)
            return {k: v for k, v in spec_dict.items() if k in known}

        sources = {}
        for name, spec in data["sources"].items():
            sources[name] = SourceSpec(name=name,
                                       **_filter_known(SourceSpec, spec, f"source {name}"))
        chimera_dict = data["chimera_v51"]
        chimera = ChimeraSpec(**_filter_known(ChimeraSpec, chimera_dict, "chimera_v51"))
        return cls(version=data["version"], sources=sources, chimera=chimera)

    def list_sources(self) -> list[str]:
        return list(self.sources.keys())

    def get_source(self, name: str) -> SourceSpec:
        if name not in self.sources:
            raise KeyError(f"unknown source: {name}; known={list(self.sources.keys())}")
        return self.sources[name]

    def list_features(self) -> list[str]:
        names = []
        for src in self.sources.values():
            names.extend(src.output_feature_names())
        return names

    def get_chimera_join_order(self) -> list[str]:
        return list(self.chimera.sources_to_join)

    def get_expected_v51_features(self) -> int:
        return self.chimera.expected_chimera_v51_total_features

    def validate_against_disk(self) -> list[str]:
        """Return list of warnings/errors about source files vs registry spec."""
        msgs: list[str] = []
        for name, src in self.sources.items():
            fp = src.absolute_path()
            if not fp.exists():
                msgs.append(f"[MISSING] {name}: {fp}")
                continue
            try:
                import polars as pl
                schema = pl.read_parquet_schema(fp)
                cols = set(schema.keys())
                if src.date_col not in cols:
                    msgs.append(f"[BAD_DATE_COL] {name}: '{src.date_col}' not in {sorted(cols)}")
                if (src.layout == "per_asset" and src.asset_col
                        and src.asset_col not in cols):
                    msgs.append(f"[MISSING_ASSET_COL] {name}: '{src.asset_col}' not in {sorted(cols)}")
                if src.layout == "wide_per_asset" and src.wide_pattern:
                    pat = re.compile(src.wide_pattern)
                    matches = [c for c in cols if pat.match(c)]
                    if not matches:
                        msgs.append(f"[NO_WIDE_MATCH] {name}: pattern '{src.wide_pattern}' matches 0 columns")
                missing = [f for f in src.features if f not in cols]
                if missing:
                    msgs.append(f"[MISSING_FEATURES] {name}: {missing}")
            except Exception as e:
                msgs.append(f"[READ_ERR] {name}: {e}")
        return msgs


def main() -> None:
    """CLI: print registry summary + validate."""
    reg = FeatureRegistry.load()
    print(f"FeatureRegistry v{reg.version}")
    print(f"  {len(reg.sources)} sources")
    print(f"  {len(reg.list_features())} feature names total")
    print(f"  expected chimera v51 total features: {reg.get_expected_v51_features()}")
    print(f"\nSources (in join order):")
    for name in reg.get_chimera_join_order():
        src = reg.get_source(name)
        print(f"  {name:25s} layout={src.layout:18s} prefix={src.prefix:6s} features={len(src.features)}")
    print(f"\nValidation against disk:")
    msgs = reg.validate_against_disk()
    if not msgs:
        print("  OK -- all source files present and schemas match.")
    else:
        for m in msgs:
            print(f"  {m}")


if __name__ == "__main__":
    main()
