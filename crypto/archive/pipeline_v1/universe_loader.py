"""Universe loader — single point for U10/U50/U100 access.

Replaces hard-coded UNIVERSE_10/UNIVERSE_50_LIQUID/UNIVERSE_100_TARGET in
src/strategy/universe.py. The new layer reads from config/universes/*.yaml,
which is the canonical declarative spec.

Public API:
    universes = UniverseLoader.load()
    universes.list("u50")               # ['BTCUSDT', 'ETHUSDT', ...]
    universes.dna_for("BTCUSDT")        # 'BLUE'
    universes.is_in("BTCUSDT", "u50")   # True
    universes.position_cap("BTCUSDT", "u50")    # 0.08
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_PATH = PROJECT_ROOT / "config" / "universes_index.yaml"
UNIVERSES_DIR = PROJECT_ROOT / "config" / "universes"


@dataclass
class AssetSpec:
    symbol: str
    dna: str
    pos_cap: float
    kelly_frac: float
    tier: str = ""                          # legacy hand-picked tier (A/B/C)
    status: str = "ready"
    # CDAP-driven liquidity classification (populated by
    # scripts/screen_universe_by_liquidity.py / apply_liquidity_tiers.py).
    # Values: TIER_B, TIER_C, EVENT_ONLY, DROP, DROP_NO_DATA, "" (unset)
    liquidity_tier:        str = ""
    # 30-day median daily dollar volume on Binance (USD).
    median_dollar_vol_30d: float = 0.0


# Tier -> cap multiplier policy. Multiplied against the asset's `pos_cap` to
# get the effective per-asset position cap. Use via
# UniverseLoader.effective_pos_cap(symbol).
TIER_CAP_MULTIPLIER = {
    "TIER_B":       1.00,    # >= $50M/day: full size
    "TIER_C":       0.50,    # $10-50M/day: half size (capacity-aware)
    "EVENT_ONLY":   0.25,    # $1-10M/day: event sleeve only, quarter size
    "DROP":         0.00,    # < $1M/day: do not deploy continuous capital
    "DROP_NO_DATA": 0.00,    # missing data: same as drop
    "":             1.00,    # legacy / unset: defer to pos_cap as-is
}


@dataclass
class Universe:
    name: str
    description: str
    n_assets: int
    parent: str | None = None
    assets: list[AssetSpec] = field(default_factory=list)

    def list_symbols(self) -> list[str]:
        return [a.symbol for a in (self.assets or [])]

    def find(self, symbol: str) -> AssetSpec | None:
        for a in self.assets or []:
            if a.symbol == symbol:
                return a
        return None


class UniverseLoader:
    def __init__(self, universes: dict[str, Universe], default: str = "u50",
                 dna_buckets: dict | None = None):
        self.universes = universes
        self.default = default
        self.dna_buckets = dna_buckets or {}

    @classmethod
    def load(cls, index_path: Path = INDEX_PATH) -> "UniverseLoader":
        with open(index_path) as f:
            idx = yaml.safe_load(f)
        universes = {}
        for u_name, rel_path in idx["universes"].items():
            full_path = PROJECT_ROOT / rel_path
            with open(full_path) as f:
                spec = yaml.safe_load(f)
            assets_raw = spec.get("assets") or []
            extra = spec.get("extra_assets") or []
            inherit = spec.get("inherit_from")
            inherit_assets = []
            if inherit:
                if inherit in universes:
                    inherit_assets = list(universes[inherit].assets or [])
                else:
                    # inherit_from must reference an EARLIER-defined universe
                    # (dicts preserve YAML order). A forward/typo reference would
                    # otherwise silently yield zero inherited assets.
                    print(f"[universe_loader] WARN {u_name}: inherit_from="
                          f"{inherit!r} not yet loaded (define it earlier in "
                          f"universes_index.yaml); inheriting 0 assets", flush=True)
            all_assets = inherit_assets + [
                AssetSpec(**a) for a in (assets_raw + extra)
            ]
            # Deduplicate by symbol (later overrides earlier — extra_assets win over inherited)
            seen = {}
            for a in all_assets:
                seen[a.symbol] = a
            universes[u_name] = Universe(
                name=spec["name"],
                description=spec.get("description", ""),
                n_assets=len(seen),
                parent=spec.get("parent"),
                assets=list(seen.values()),
            )
        return cls(
            universes=universes,
            default=idx.get("default", "u50"),
            dna_buckets=idx.get("dna_buckets", {}),
        )

    def _resolve(self, name: str | None) -> Universe:
        """Resolve a universe by name (or default), with a helpful error."""
        key = name or self.default
        if key not in self.universes:
            raise KeyError(
                f"unknown universe {key!r}; valid: {sorted(self.universes)}")
        return self.universes[key]

    def list(self, name: str | None = None) -> list[str]:
        return self._resolve(name).list_symbols()

    def get(self, name: str | None = None) -> Universe:
        return self._resolve(name)

    def is_in(self, symbol: str, universe: str) -> bool:
        return symbol in self.list(universe)

    def dna_for(self, symbol: str, universe: str | None = None) -> str:
        # Look in the specified universe (default = u50, which has full DNA coverage)
        u = self._resolve(universe)
        spec = u.find(symbol)
        return spec.dna if spec else "UNKNOWN"

    def position_cap(self, symbol: str, universe: str | None = None) -> float:
        u = self._resolve(universe)
        spec = u.find(symbol)
        return spec.pos_cap if spec else 0.0

    def kelly_frac(self, symbol: str, universe: str | None = None) -> float:
        u = self._resolve(universe)
        spec = u.find(symbol)
        return spec.kelly_frac if spec else 0.0

    def liquidity_tier(self, symbol: str, universe: str | None = None) -> str:
        """Measured liquidity tier (TIER_B / TIER_C / EVENT_ONLY / DROP / "")."""
        u = self._resolve(universe)
        spec = u.find(symbol)
        return getattr(spec, "liquidity_tier", "") if spec else ""

    def effective_pos_cap(self, symbol: str, universe: str | None = None) -> float:
        """pos_cap * TIER_CAP_MULTIPLIER[liquidity_tier].

        EVENT_ONLY assets get 25% size; DROP get 0; un-tiered legacy get 1.0
        (defer to pos_cap as authored).
        """
        u = self._resolve(universe)
        spec = u.find(symbol)
        if spec is None:
            return 0.0
        tier = getattr(spec, "liquidity_tier", "") or ""
        mult = TIER_CAP_MULTIPLIER.get(tier, 1.0)
        return float(spec.pos_cap) * float(mult)

    def median_dollar_vol(self, symbol: str, universe: str | None = None) -> float:
        u = self._resolve(universe)
        spec = u.find(symbol)
        return getattr(spec, "median_dollar_vol_30d", 0.0) if spec else 0.0

    def filter_by_tier(self, tiers: list[str], universe: str | None = None) -> list[str]:
        """Return symbols in `universe` whose liquidity_tier is in `tiers`.

        Examples:
            loader.filter_by_tier(["TIER_B"], "u100")            # continuous-only
            loader.filter_by_tier(["TIER_B", "TIER_C"], "u100")  # cont + capped
            loader.filter_by_tier(["EVENT_ONLY"], "u100")         # hunter sleeve
        """
        u = self._resolve(universe)
        out = []
        target = set(t.upper() for t in tiers)
        for spec in (u.assets or []):
            tier = (getattr(spec, "liquidity_tier", "") or "").upper()
            if tier in target:
                out.append(spec.symbol)
        return out


def main() -> None:
    loader = UniverseLoader.load()
    print(f"Universes loaded: {list(loader.universes.keys())}")
    for name, u in loader.universes.items():
        print(f"\n{name}: {u.description}")
        print(f"  count: {u.n_assets}")
        symbols = u.list_symbols()
        print(f"  first 5: {symbols[:5]}")
        print(f"  last 5: {symbols[-5:]}")
    print(f"\ndefault: {loader.default}")
    # Sanity:
    print(f"\nBTC in u10: {loader.is_in('BTCUSDT', 'u10')}")
    print(f"BTC dna: {loader.dna_for('BTCUSDT')}")
    print(f"BTC pos_cap: {loader.position_cap('BTCUSDT')}")
    print(f"DOGE in u10: {loader.is_in('DOGEUSDT', 'u10')}")
    print(f"DOGE in u50: {loader.is_in('DOGEUSDT', 'u50')}")
    print(f"FLOKI in u100: {loader.is_in('FLOKIUSDT', 'u100')}")


if __name__ == "__main__":
    main()
