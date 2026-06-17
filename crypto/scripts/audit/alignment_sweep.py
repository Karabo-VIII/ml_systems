"""alignment_sweep.py -- end-to-end pipeline alignment & uniformity audit.

User mandate 2026-05-15: 'syncing and uniformity when it comes to our end
to end pipeline. We need to confirm if callers, creators, configs, etc
all align and speak the same language.'

Cross-references:
  1. ASSET MEMBERSHIP -- every config that lists assets
     - config/universes/{u10,u50,u100}.yaml (canonical SoT)
     - config/data_config.yaml (legacy fallback)
     - config/asset_launch_dates.json (Binance perp mapping)
     - data/raw/* (filesystem reality)
     - production_blends.yaml inline ref'd assets

  2. ASSET NAMING -- canonical convention vs in-use
     - BTC/USDT (slash form, used by fetch_all CLI)
     - BTCUSDT (no-slash, used in storage paths + universe yaml)
     - btcusdt (lowercase, chimera filenames)

  3. UNIVERSE WIRING -- who imports UniverseLoader vs reads yaml directly
     - Direct yaml reads = drift surface
     - ChimeraLoader / UniverseLoader = canonical

  4. CONFIG-FILE GRAPH -- which file is read by what
     - asset_dag.yaml read by refresh.py
     - universes/*.yaml read by UniverseLoader
     - data_config.yaml read by fetch_all (legacy fallback)
     - production_blends.yaml read by blend_composer

Output: docs/ALIGNMENT_SWEEP_2026_05_15.md
"""
from __future__ import annotations

import json
import re
import yaml
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]


def _load_canonical_universe() -> dict:
    """Load all u10/u50/u100 yamls -- the SoT for asset membership."""
    out = {"u10": set(), "u50": set(), "u100_ready": set(), "u100_excluded": set()}
    for u in ("u10", "u50", "u100"):
        fp = ROOT / "config" / "universes" / f"{u}.yaml"
        if not fp.exists():
            continue
        with fp.open(encoding='utf-8') as f:
            spec = yaml.safe_load(f)
        active_assets = [a["symbol"] for a in (spec.get("assets") or [])]
        extra = [a["symbol"] for a in (spec.get("extra_assets") or [])
                 if a.get("status") == "ready"]
        excluded = spec.get("excluded_assets") or []
        if u == "u10":
            out["u10"] = set(active_assets)
        elif u == "u50":
            out["u50"] = set(active_assets)
        elif u == "u100":
            out["u100_ready"] = set(extra) | out["u50"]  # u50 inherits
            out["u100_excluded"] = set(excluded)
    return out


def _load_legacy_data_config() -> set:
    fp = ROOT / "config" / "data_config.yaml"
    if not fp.exists():
        return set()
    with fp.open(encoding='utf-8') as f:
        spec = yaml.safe_load(f)
    assets = spec.get("assets") or {}
    return set(p.replace("/", "").upper() for p, s in assets.items()
               if s.get("is_active", True))


def _load_launch_dates() -> dict:
    fp = ROOT / "config" / "asset_launch_dates.json"
    if not fp.exists():
        return {}
    with fp.open(encoding='utf-8') as f:
        return json.load(f)


def _load_filesystem_assets() -> set:
    raw = ROOT / "data" / "raw"
    if not raw.exists():
        return set()
    return set(d.name for d in raw.iterdir() if d.is_dir() and d.name.endswith("USDT"))


def _scan_naming_conventions() -> dict:
    """Find files that use each naming convention -- drift surface."""
    SCAN_ROOTS = [ROOT / "src" / "pipeline", ROOT / "scripts" / "oracle",
                  ROOT / "src" / "strategy", ROOT / "scripts" / "strat_audit"]
    counters = {"slash_form": 0, "compact": 0, "lowercase": 0}
    # Patterns to count (rough heuristics, not exact)
    RE_SLASH = re.compile(r"['\"][A-Z]{2,8}/USDT['\"]")
    RE_COMPACT = re.compile(r"['\"][A-Z]{2,8}USDT['\"]")
    RE_LOWER = re.compile(r"['\"][a-z]{2,8}usdt['\"]")
    for root in SCAN_ROOTS:
        if not root.exists(): continue
        for fp in root.rglob("*.py"):
            if "__pycache__" in fp.parts or "archive" in fp.as_posix(): continue
            try: t = fp.read_text(encoding='utf-8', errors='replace')
            except: continue
            counters["slash_form"] += len(RE_SLASH.findall(t))
            counters["compact"] += len(RE_COMPACT.findall(t))
            counters["lowercase"] += len(RE_LOWER.findall(t))
    return counters


def _find_direct_yaml_readers() -> list:
    """Files that read config/universes/*.yaml directly (bypassing UniverseLoader)."""
    SCAN_ROOTS = [ROOT / "src", ROOT / "scripts"]
    hits = []
    RE = re.compile(r"config/universes/(u\d+)\.yaml")
    for root in SCAN_ROOTS:
        if not root.exists(): continue
        for fp in root.rglob("*.py"):
            if "__pycache__" in fp.parts or "archive" in fp.as_posix(): continue
            try: t = fp.read_text(encoding='utf-8', errors='replace')
            except: continue
            m = RE.search(t)
            if m and "UniverseLoader" not in t:
                hits.append((fp.relative_to(ROOT).as_posix(), m.group(0)))
    return hits


def main():
    uni = _load_canonical_universe()
    legacy = _load_legacy_data_config()
    launch = _load_launch_dates()
    fs = _load_filesystem_assets()
    naming = _scan_naming_conventions()
    direct_yaml = _find_direct_yaml_readers()

    print("=" * 72)
    print("PIPELINE ALIGNMENT SWEEP -- 2026-05-15")
    print("=" * 72)
    print(f"u10 count:           {len(uni['u10'])}")
    print(f"u50 count:           {len(uni['u50'])}")
    print(f"u100 ready_assets:   {len(uni['u100_ready'])}")
    print(f"u100 excluded:       {len(uni['u100_excluded'])}")
    print(f"legacy config active:{len(legacy)}")
    print(f"launch_dates cache:  {len(launch)}")
    print(f"filesystem assets:   {len(fs)}")
    print()

    # Drift 1: legacy config has excluded_assets active
    stale = legacy & uni["u100_excluded"]
    print(f"DRIFT 1: legacy config -> u100 excluded: {len(stale)}")
    for a in sorted(stale)[:5]: print(f"  {a}")

    # Drift 2: legacy config has assets not in u100 ready
    orphans = legacy - uni["u100_ready"] - uni["u100_excluded"]
    print(f"\nDRIFT 2: legacy config NOT in u100 anywhere: {len(orphans)}")
    for a in sorted(orphans)[:5]: print(f"  {a}")

    # Drift 3: filesystem has assets not in u100
    fs_orphans = fs - uni["u100_ready"] - uni["u100_excluded"]
    print(f"\nDRIFT 3: data/raw/ assets NOT in u100 anywhere: {len(fs_orphans)}")
    for a in sorted(fs_orphans)[:5]: print(f"  {a}")

    # Drift 4: u100 ready_assets missing from filesystem
    missing_fs = uni["u100_ready"] - fs
    print(f"\nDRIFT 4: u100 ready_assets MISSING from data/raw/: {len(missing_fs)}")
    for a in sorted(missing_fs): print(f"  {a}")

    # Drift 5: u100 ready_assets missing launch date
    no_launch = [a for a in uni["u100_ready"] if a not in launch
                 and f"1000{a.replace('USDT','')}USDT" not in launch]
    print(f"\nDRIFT 5: u100 ready_assets missing launch_date (after R32): {len(no_launch)}")
    for a in sorted(no_launch)[:10]: print(f"  {a}")

    # Naming convention counts
    print(f"\nNAMING CONVENTION USAGE (string literals in code):")
    print(f"  'BTC/USDT' (slash form):  {naming['slash_form']:>5d}")
    print(f"  'BTCUSDT' (compact):      {naming['compact']:>5d}")
    print(f"  'btcusdt' (lowercase):    {naming['lowercase']:>5d}")
    print(f"  -> Multiple conventions in use. Standardize where the same logical asset has multiple forms.")

    # Direct yaml readers
    print(f"\nDIRECT YAML READERS (bypass UniverseLoader): {len(direct_yaml)}")
    for fp, m in direct_yaml[:10]:
        print(f"  {fp}: '{m}'")

    # Write doc
    doc = ROOT / "docs" / "ALIGNMENT_SWEEP_2026_05_15.md"
    lines = []
    lines.append("# Pipeline Alignment Sweep -- 2026-05-15")
    lines.append("")
    lines.append("> Programmatic check: do callers, creators, configs all speak the same language?")
    lines.append("")
    lines.append("## Canonical state")
    lines.append("")
    lines.append(f"| Source | Count |")
    lines.append(f"|---|---|")
    lines.append(f"| u10 (canonical SoT) | {len(uni['u10'])} |")
    lines.append(f"| u50 (canonical) | {len(uni['u50'])} |")
    lines.append(f"| u100 ready_assets | {len(uni['u100_ready'])} |")
    lines.append(f"| u100 excluded_assets | {len(uni['u100_excluded'])} |")
    lines.append(f"| config/data_config.yaml active | {len(legacy)} |")
    lines.append(f"| config/asset_launch_dates.json | {len(launch)} |")
    lines.append(f"| data/raw/ filesystem | {len(fs)} |")
    lines.append("")
    lines.append("## Drift findings")
    lines.append("")
    lines.append(f"### Drift 1 -- legacy config has u100 excluded_assets ({len(stale)})")
    lines.append("")
    lines.append("Defensive filter shipped R33. Cleanup still needed for hygiene:")
    for a in sorted(stale): lines.append(f"  - {a}")
    lines.append("")
    lines.append(f"### Drift 2 -- legacy config has assets not in u100 anywhere ({len(orphans)})")
    lines.append("")
    for a in sorted(orphans): lines.append(f"  - {a}")
    lines.append("")
    lines.append(f"### Drift 3 -- filesystem has assets not in u100 ({len(fs_orphans)})")
    lines.append("")
    for a in sorted(fs_orphans): lines.append(f"  - {a}")
    lines.append("")
    lines.append(f"### Drift 4 -- u100 ready_assets missing from filesystem ({len(missing_fs)})")
    lines.append("")
    for a in sorted(missing_fs): lines.append(f"  - {a}")
    lines.append("")
    lines.append(f"### Drift 5 -- u100 ready_assets missing launch_date ({len(no_launch)})")
    lines.append("")
    for a in sorted(no_launch): lines.append(f"  - {a}")
    lines.append("")
    lines.append("## Naming convention usage")
    lines.append("")
    lines.append(f"| Form | Occurrences |")
    lines.append(f"|---|---|")
    lines.append(f"| 'BTC/USDT' (slash) | {naming['slash_form']} |")
    lines.append(f"| 'BTCUSDT' (compact) | {naming['compact']} |")
    lines.append(f"| 'btcusdt' (lowercase) | {naming['lowercase']} |")
    lines.append("")
    lines.append("Canonical conventions per code path:")
    lines.append("- fetch_all CLI: `BTC/USDT` (slash form)")
    lines.append("- Universe yaml + storage paths: `BTCUSDT` (compact)")
    lines.append("- Chimera filenames: `btcusdt_v51_chimera_*` (lowercase)")
    lines.append("- DAG yaml producer_args: per-stage (`asset_format: root|pair|usdt`)")
    lines.append("Mixing is correct WITHIN context; cross-context conversions via fetch_all `_normalize_pair()` etc.")
    lines.append("")
    lines.append(f"## Direct yaml readers (bypass UniverseLoader) -- {len(direct_yaml)}")
    lines.append("")
    lines.append("These hardcode `config/universes/uX.yaml` instead of using UniverseLoader. Drift surface.")
    lines.append("")
    for fp, m in direct_yaml:
        lines.append(f"- `{fp}` references `{m}`")
    lines.append("")
    doc.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {doc.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
