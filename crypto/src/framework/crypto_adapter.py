"""CryptoAdapter -- the crypto realization of the MarketAdapter contract (src/framework/adapter.py).

This is the template a stocks/FX adapter copies: it wraps the project's EXISTING crypto apparatus (chimera loader,
universe yamls, taker cost, feature_map families, chimera data manifests) behind the generic 5-method interface, so
the solutioning pipeline runs on crypto WITHOUT any crypto-specific code leaking into the generic stages. Nothing is
re-implemented -- it delegates. No emoji (cp1252).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

CADENCES = ["1d", "4h", "1h", "30m", "15m", "dollar", "dib"]


class CryptoCostModel:
    """Honest crypto round-trip cost. Taker 0.0024 is the project's deploy baseline (maker p_fill 0.21-0.40 is too
    unreliable to assume -- see config/maker_cost_calibration.yaml + CLAUDE.md MakerCostModel invariants)."""
    def __init__(self):
        try:
            from strat.candidate_gate import TAKER_COST_RT
            self.taker = float(TAKER_COST_RT)
        except Exception:
            self.taker = 0.0024

    def round_trip(self, symbol: str, side: str = "long", notional: float = 0.0, venue: str | None = None) -> float:
        return self.taker  # per-asset/venue refinement (bucketed maker calibration) is a follow-up


class CryptoAdapter:
    """Implements the MarketAdapter Protocol for crypto by delegating to the existing apparatus."""
    market = "crypto"

    def __init__(self):
        self._loader = None
        self._cost = CryptoCostModel()

    def universe(self, tier: str = "u100") -> Sequence[str]:
        import yaml
        p = ROOT / "config" / "universes" / f"{tier}.yaml"
        if not p.exists():
            p = ROOT / "config" / "universes" / "u100.yaml"
        spec = yaml.safe_load(p.read_text(encoding="utf-8"))
        return [a["symbol"] for a in spec.get("assets", [])]

    def load(self, symbol: str, cadence: str, features: Sequence[str] | None = None):
        if self._loader is None:
            from pipeline.chimera_loader import ChimeraLoader
            self._loader = ChimeraLoader()
        sym = symbol.upper() if symbol.upper().endswith("USDT") else symbol.upper() + "USDT"
        return self._loader.load(sym, cadence=cadence, features=list(features) if features else None)

    def cost_model(self) -> CryptoCostModel:
        return self._cost

    def cadences(self) -> Sequence[str]:
        return list(CADENCES)

    def feature_families(self) -> dict:
        from narrate import feature_map as fm
        groups = fm.group_columns(list(fm.FEATURES))  # {family: [cols]} in canonical order
        return {k: v for k, v in groups.items() if v}

    def data_snapshot_id(self, symbol: str | None = None) -> str:
        """A content id for the data version, for pipeline lineage (--data-ref). Hash of the chimera manifest(s)."""
        manifests = ROOT / "data" / "manifests"
        if symbol:
            sym = symbol.upper() if symbol.upper().endswith("USDT") else symbol.upper() + "USDT"
            mf = manifests / f"v51_{sym}.json"
            files = [mf] if mf.exists() else []
        else:
            files = sorted(manifests.glob("v51_*.json"))
        if not files:
            return "no-manifest"
        h = hashlib.sha256()
        for f in files:
            try:
                h.update(f.read_bytes())
            except Exception:
                pass
        return f"chimera_v51:{len(files)}f:{h.hexdigest()[:12]}"


if __name__ == "__main__":
    a = CryptoAdapter()
    print("market           :", a.market)
    print("universe(u10)    :", a.universe("u10"))
    print("cadences         :", a.cadences())
    print("cost round_trip  :", a.cost_model().round_trip("BTCUSDT"))
    ff = a.feature_families()
    print("feature_families :", {k: len(v) for k, v in ff.items()})
    print("data_snapshot_id :", a.data_snapshot_id("BTCUSDT"))
