"""Forex universe loader — single point for u7/u28/u52 access.

Reads the declarative specs at config/universes/*.yaml (source of truth) via
config/universes_index.yaml. Analogous to the crypto UniverseLoader, adapted
for FX pair metadata (pip size, price divisor, liquidity tier, BIS turnover).

Public API:
    U = UniverseLoader.load()
    U.list("u7")                 # ['EURUSD', 'USDJPY', ...]
    U.pip("USDJPY")              # 0.01
    U.price_divisor("EURUSD")    # 100000.0   (Dukascopy int-price scale)
    U.dna_for("EURUSD")          # 'MAJOR'
    U.is_in("EURUSD", "u7")      # True
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_PATH = PROJECT_ROOT / "config" / "universes_index.yaml"
UNIVERSES_DIR = PROJECT_ROOT / "config" / "universes"

__contract__ = {
    "kind": "framework_helper",
    "stage": "universe_loader",
    "inputs": {"config": "config/universes_index.yaml + config/universes/*.yaml"},
    "outputs": {"api": "list/pip/price_divisor/dna_for/pos_cap per pair"},
    "invariants": {"yaml_is_source_of_truth": True, "tier_inheritance_u7_u28_u52": True},
    "rationale": "Single point for u7/u28/u52 pair metadata; no hardcoded pair lists.",
}


@dataclass
class PairSpec:
    symbol: str                       # EURUSD (flat, UPPER)
    dna: str = "MAJOR"                # MAJOR / CROSS / EXOTIC / METAL
    pip: float = 0.0001              # 0.0001 non-JPY, 0.01 JPY/metals
    price_divisor: float = 100000.0  # Dukascopy integer-price scale (10^decimals)
    pos_cap: float = 0.10            # max fraction of book per pair
    tier: str = "A"                  # liquidity tier A(deepest)/B/C
    turnover_share: float = 0.0      # BIS Triennial % of global FX turnover (0 if n/a)


@dataclass
class Universe:
    name: str
    description: str
    n_pairs: int
    parent: str | None = None
    pairs: list[PairSpec] = field(default_factory=list)

    def list_symbols(self) -> list[str]:
        return [p.symbol for p in (self.pairs or [])]

    def find(self, symbol: str) -> PairSpec | None:
        sym = symbol.upper().replace("/", "").replace("_", "")
        for p in self.pairs or []:
            if p.symbol == sym:
                return p
        return None


class UniverseLoader:
    def __init__(self, universes: dict[str, Universe], default: str = "u7"):
        self.universes = universes
        self.default = default

    @classmethod
    def load(cls, index_path: Path = INDEX_PATH) -> "UniverseLoader":
        with open(index_path) as f:
            idx = yaml.safe_load(f)
        universes: dict[str, Universe] = {}
        for u_name, rel_path in idx["universes"].items():
            full_path = PROJECT_ROOT / rel_path
            with open(full_path) as f:
                spec = yaml.safe_load(f)
            pairs_raw = spec.get("pairs") or []
            extra = spec.get("extra_pairs") or []
            inherit = spec.get("inherit_from")
            inherit_pairs: list[PairSpec] = []
            if inherit:
                if inherit in universes:
                    inherit_pairs = list(universes[inherit].pairs or [])
                else:
                    print(f"[universe_loader] WARN {u_name}: inherit_from="
                          f"{inherit!r} not yet loaded (define it earlier in "
                          f"universes_index.yaml); inheriting 0 pairs", flush=True)
            all_pairs = inherit_pairs + [PairSpec(**p) for p in (pairs_raw + extra)]
            # Dedup by symbol (later overrides earlier -> extra_pairs win).
            seen: dict[str, PairSpec] = {}
            for p in all_pairs:
                seen[p.symbol] = p
            universes[u_name] = Universe(
                name=spec["name"],
                description=spec.get("description", ""),
                n_pairs=len(seen),
                parent=spec.get("parent"),
                pairs=list(seen.values()),
            )
        return cls(universes=universes, default=idx.get("default", "u7"))

    def _resolve(self, name: str | None) -> Universe:
        key = name or self.default
        if key not in self.universes:
            raise KeyError(f"unknown universe {key!r}; valid: {sorted(self.universes)}")
        return self.universes[key]

    def list(self, name: str | None = None) -> list[str]:
        return self._resolve(name).list_symbols()

    def get(self, name: str | None = None) -> Universe:
        return self._resolve(name)

    def is_in(self, symbol: str, universe: str) -> bool:
        return self._resolve(universe).find(symbol) is not None

    def _spec(self, symbol: str, universe: str | None) -> PairSpec | None:
        # Search the named universe first, then any universe (metadata is global).
        spec = self._resolve(universe).find(symbol)
        if spec is not None:
            return spec
        for u in self.universes.values():
            s = u.find(symbol)
            if s is not None:
                return s
        return None

    def pip(self, symbol: str, universe: str | None = None) -> float:
        s = self._spec(symbol, universe)
        return s.pip if s else (0.01 if symbol.upper().endswith("JPY") else 0.0001)

    def price_divisor(self, symbol: str, universe: str | None = None) -> float:
        s = self._spec(symbol, universe)
        if s:
            return s.price_divisor
        return 1000.0 if symbol.upper().endswith("JPY") else 100000.0

    def dna_for(self, symbol: str, universe: str | None = None) -> str:
        s = self._spec(symbol, universe)
        return s.dna if s else "UNKNOWN"

    def pos_cap(self, symbol: str, universe: str | None = None) -> float:
        s = self._spec(symbol, universe)
        return s.pos_cap if s else 0.0


def main() -> None:
    U = UniverseLoader.load()
    print(f"Universes loaded: {list(U.universes.keys())}")
    for name, u in U.universes.items():
        syms = u.list_symbols()
        print(f"\n{name}: {u.description}")
        print(f"  count: {u.n_pairs}")
        print(f"  first 5: {syms[:5]}")
        print(f"  last 5: {syms[-5:]}")
    print(f"\ndefault: {U.default}")
    print(f"\nEURUSD in u7: {U.is_in('EURUSD', 'u7')}  pip={U.pip('EURUSD')}  "
          f"div={U.price_divisor('EURUSD')}  dna={U.dna_for('EURUSD')}")
    print(f"USDJPY pip={U.pip('USDJPY')}  div={U.price_divisor('USDJPY')}")
    print(f"XAUUSD dna={U.dna_for('XAUUSD')}  pip={U.pip('XAUUSD')}")


if __name__ == "__main__":
    main()
