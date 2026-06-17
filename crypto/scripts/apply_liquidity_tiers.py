"""
Apply screener-measured liquidity tiers to universe yamls.

Reads logs/universe_screen_<DATE>.csv (produced by
scripts/screen_universe_by_liquidity.py) and updates:
    config/universes/u10.yaml
    config/universes/u50.yaml
    config/universes/u100.yaml

For each asset listed in the yaml, adds/refreshes:
    liquidity_tier:        TIER_B / TIER_C / EVENT_ONLY / DROP / DROP_NO_DATA
    median_dollar_vol_30d: <USD>

Existing fields (dna, pos_cap, kelly_frac, status, tier) are preserved.

Default behaviour writes a `.proposed.yaml` next to each existing yaml
for review. Pass --apply to overwrite in place.

Usage:
    python scripts/apply_liquidity_tiers.py
    python scripts/apply_liquidity_tiers.py --apply
    python scripts/apply_liquidity_tiers.py --csv logs/universe_screen_20260428.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

# 2026-05-15 R35: canonical universe access via UniverseLoader (alignment).
# This file historically reads config/universes/*.yaml directly. The yaml read
# remains for fields not yet wrapped (DNA, liquidity_tier metadata); the
# UniverseLoader import below documents the canonical access point for
# asset-list enumeration. Future: migrate yaml.safe_load paths to
# UniverseLoader methods (.list, .dna_for, .liquidity_tier, .position_cap).
try:
    import sys as _r35_sys
    from pathlib import Path as _r35_Path
    _r35_SRC = _r35_Path(__file__).resolve().parent
    while _r35_SRC.name and not (_r35_SRC / "pipeline" / "universe_loader.py").exists() and _r35_SRC != _r35_SRC.parent:
        _r35_SRC = _r35_SRC.parent
    if str(_r35_SRC) not in _r35_sys.path:
        _r35_sys.path.insert(0, str(_r35_SRC))
    from pipeline.universe_loader import UniverseLoader  # noqa: F401 (canonical access marker)
except ImportError:
    UniverseLoader = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNIVERSES_DIR = PROJECT_ROOT / "config" / "universes"
LOG_DIR = PROJECT_ROOT / "logs"

YAMLS = ["u10.yaml", "u50.yaml", "u100.yaml"]


def _latest_csv() -> Path | None:
    candidates = sorted(LOG_DIR.glob("universe_screen_*.csv"))
    return candidates[-1] if candidates else None


def load_screener(csv_path: Path) -> dict:
    """Read screener output -> {SYMBOL: {tier, median_vol}}."""
    out = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            sym = row["asset"].strip().upper()
            vol_raw = (row.get("median_$vol") or row.get("median_vol") or "0").strip()
            try:
                vol = float(vol_raw) if vol_raw not in ("", "nan") else 0.0
            except ValueError:
                vol = 0.0
            out[sym] = {
                "tier":       row.get("tier", "").strip(),
                "median_vol": vol,
            }
    return out


def patch_yaml(path: Path, screener: dict, apply_in_place: bool = False) -> Path:
    """Patch a universe yaml file by injecting liquidity_tier + median_dollar_vol_30d
    into each `- { symbol: ... }` line. Preserves all other fields and comments.

    Returns the path actually written.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8")
    original = text

    # The yaml uses inline-flow-mapping per asset. Pattern:
    # - { symbol: BTCUSDT, dna: BLUE, pos_cap: 0.10, kelly_frac: 0.40, ... }
    sym_re = re.compile(
        r"^(?P<lead>\s*-\s*\{\s*symbol\s*:\s*)(?P<sym>[A-Z0-9_]+)(?P<rest>[^}]*)\}",
        re.M,
    )

    def _patch(match: re.Match) -> str:
        sym = match.group("sym").upper()
        rest = match.group("rest")
        info = screener.get(sym)
        if info is None:
            # Asset not in screener output (e.g. fetched but no chimera).
            # Mark as DROP_NO_DATA conservatively.
            tier = "DROP_NO_DATA"
            vol = 0.0
        else:
            tier = info["tier"] or "DROP_NO_DATA"
            vol = info["median_vol"]

        # Strip any pre-existing liquidity_tier / median_dollar_vol_30d
        rest = re.sub(r",\s*liquidity_tier\s*:\s*[A-Z_]+", "", rest)
        rest = re.sub(r",\s*median_dollar_vol_30d\s*:\s*[0-9_.]+", "", rest)

        # Append fresh fields
        new_rest = rest.rstrip().rstrip(",")
        new_rest += f", liquidity_tier: {tier}, median_dollar_vol_30d: {int(round(vol))}"
        return f"{match.group('lead')}{sym}{new_rest} }}"

    new_text = sym_re.sub(_patch, text)

    if new_text == original:
        print(f"[skip] {path.name}: no asset entries matched the patch regex "
              f"(file may use a different yaml style)")
        return path

    if apply_in_place:
        out_path = path
    else:
        out_path = path.with_suffix(".proposed.yaml")
    out_path.write_text(new_text, encoding="utf-8")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None,
                    help="screener csv path (default: latest in logs/)")
    ap.add_argument("--apply", action="store_true",
                    help="overwrite yamls in place (default: write .proposed.yaml siblings)")
    args = ap.parse_args()

    csv_path = Path(args.csv) if args.csv else _latest_csv()
    if csv_path is None or not csv_path.exists():
        print("[error] no screener csv found in logs/. "
              "Run scripts/screen_universe_by_liquidity.py first.", file=sys.stderr)
        return 1
    print(f"Loading screener: {csv_path.relative_to(PROJECT_ROOT)}")

    screener = load_screener(csv_path)
    print(f"  {len(screener)} assets in screener output")
    print()

    for yname in YAMLS:
        path = UNIVERSES_DIR / yname
        if not path.exists():
            print(f"[skip] {yname}: not found at {path}")
            continue
        out = patch_yaml(path, screener, apply_in_place=args.apply)
        print(f"  {yname:<14} -> {out.relative_to(PROJECT_ROOT)}")

    print()
    if not args.apply:
        print("Wrote .proposed.yaml siblings. Diff and review, then re-run with --apply.")
    else:
        print("Applied in place. Verify universe_loader still loads cleanly:")
        print("  python src/pipeline/universe_loader.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
